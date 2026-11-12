import asyncio
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from unittest import mock

import nexusrpc
import nexusrpc.handler
import pytest
from nexusrpc.handler._core import ServiceHandler

import temporalio.nexus
import temporalio.nexus._operation_context
import temporalio.nexus._token
import temporalio.worker._nexus
from temporalio import activity, workflow
from temporalio.client import (
    ActivityExecutionStatus,
    Client,
    NexusClient,
    NexusOperationHandle,
    WorkflowFailureError,
)
from temporalio.common import NexusOperationExecutionStatus
from temporalio.exceptions import NexusOperationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import (
    ExecuteNexusOperationCancelInput,
    ExecuteNexusOperationStartInput,
    Interceptor,
    NexusOperationInboundInterceptor,
    Worker,
)
from tests.helpers import assert_eventually
from tests.helpers.nexus import make_nexus_endpoint_name


@workflow.defn
class WaitingWorkflow:
    @workflow.run
    async def run(self) -> None:
        await workflow.wait_condition(lambda: False)


@workflow.defn
class EchoWorkflow:
    @workflow.run
    async def run(self, value: str) -> str:
        return value


@activity.defn
async def wait_for_cancel_activity() -> None:
    # Heartbeat so the activity receives cancellation.
    while True:
        await asyncio.sleep(0.3)
        activity.heartbeat()


@nexusrpc.handler.service_handler(name="tools")
class CatchAllService:
    @nexusrpc.handler.sync_operation
    async def list_tools(
        self, ctx: nexusrpc.handler.StartOperationContext, input: None
    ) -> str:
        return "manifest"

    @temporalio.nexus.temporal_operation(dynamic=True)
    async def forward(
        self,
        ctx: temporalio.nexus.TemporalStartOperationContext,
        client: temporalio.nexus.TemporalNexusClient,
        input: nexusrpc.LazyValue,
    ) -> temporalio.nexus.TemporalOperationResult[Any]:
        arguments = await input.consume(as_type=dict)
        if ctx.operation == "async":
            return temporalio.nexus.TemporalOperationResult.async_token(
                "operation-token"
            )
        if ctx.operation == "reject":
            raise nexusrpc.HandlerError(
                "Unknown tool", type=nexusrpc.HandlerErrorType.NOT_FOUND
            )
        return temporalio.nexus.TemporalOperationResult.sync(
            (ctx.operation, ctx.request_id, arguments)
        )


@pytest.mark.parametrize("name", ["get_weather", "async", "reject"])
async def test_dynamic_temporal_decorator(name: str):
    service = ServiceHandler.from_user_instance(CatchAllService())
    assert set(service.operation_handlers) == {"list_tools"}
    assert service.get_operation_handler("list_tools") is not service.dynamic_handler
    handler = service.get_operation_handler(name)
    assert isinstance(handler, temporalio.nexus.TemporalOperationHandler)
    assert handler is service.dynamic_handler
    ctx = temporalio.nexus.TemporalStartOperationContext(
        service="tools",
        operation=name,
        headers={},
        request_id="request-id",
        task_cancellation=temporalio.worker._nexus._NexusTaskCancellation(),
    )
    serializer = mock.Mock()
    serializer.deserialize = mock.AsyncMock(return_value={"city": "Boston"})
    raw = nexusrpc.LazyValue(serializer=serializer, headers={})
    if name == "reject":
        with pytest.raises(nexusrpc.HandlerError) as error:
            await handler.start_operation(ctx, mock.Mock(), raw)
        assert error.value.type == nexusrpc.HandlerErrorType.NOT_FOUND
    else:
        result = await handler.start_operation(ctx, mock.Mock(), raw)
        if name == "async":
            nexus_result = result._to_nexus_result()
            assert isinstance(nexus_result, nexusrpc.handler.StartOperationResultAsync)
            assert nexus_result.token == "operation-token"
        else:
            assert result.value == (name, "request-id", {"city": "Boston"})
    assert serializer.deserialize.call_args.kwargs["as_type"] is dict


def test_dynamic_temporal_decorator_rejects_name():
    with pytest.raises(ValueError, match="cannot specify name"):
        temporalio.nexus.temporal_operation(name="named", dynamic=True)


def test_dynamic_temporal_decorator_requires_raw_input():
    with pytest.raises(TypeError, match="LazyValue"):

        @temporalio.nexus.temporal_operation(dynamic=True)
        async def decoded(
            self,
            ctx: temporalio.nexus.TemporalStartOperationContext,
            client: temporalio.nexus.TemporalNexusClient,
            input: dict,
        ) -> temporalio.nexus.TemporalOperationResult[Any]:
            return temporalio.nexus.TemporalOperationResult.sync(input)


def test_second_dynamic_temporal_operation_fails_registration():
    with pytest.raises(RuntimeError, match="more than one dynamic"):

        @nexusrpc.handler.service_handler(name="duplicate")
        class _Duplicate(CatchAllService):
            @temporalio.nexus.temporal_operation(dynamic=True)
            async def other(
                self,
                ctx: temporalio.nexus.TemporalStartOperationContext,
                client: temporalio.nexus.TemporalNexusClient,
                input: nexusrpc.LazyValue,
            ) -> temporalio.nexus.TemporalOperationResult[Any]:
                raise NotImplementedError

        _ = _Duplicate


@nexusrpc.handler.service_handler(name="tools")
class RejectingService:
    @temporalio.nexus.temporal_operation(dynamic=True)
    async def forward(
        self,
        ctx: temporalio.nexus.TemporalStartOperationContext,
        client: temporalio.nexus.TemporalNexusClient,
        input: nexusrpc.LazyValue,
    ) -> temporalio.nexus.TemporalOperationResult[Any]:
        raise nexusrpc.HandlerError(
            "Unknown tool", type=nexusrpc.HandlerErrorType.NOT_FOUND
        )


@pytest.mark.parametrize("kind", [1, 2, 3])
async def test_dynamic_cancel_uses_token_handler(kind: int):
    client = mock.MagicMock(namespace="default")
    client.get_workflow_handle.return_value.cancel = mock.AsyncMock()
    client.get_activity_handle.return_value.cancel = mock.AsyncMock()
    ctx = nexusrpc.handler.CancelOperationContext(
        service="tools",
        operation="removed",
        headers={},
        task_cancellation=temporalio.worker._nexus._NexusTaskCancellation(),
    )
    context = temporalio.nexus._operation_context._TemporalCancelOperationContext(
        nexus_context=ctx,
        client=client,
        info=lambda: temporalio.nexus.Info(
            endpoint="endpoint", namespace="default", task_queue="tq"
        ),
        _runtime_metric_meter=mock.MagicMock(),
        _worker_shutdown_event=mock.MagicMock(),
    )
    context_token = (
        temporalio.nexus._operation_context._temporal_cancel_operation_context.set(
            context
        )
    )
    try:
        handler = nexusrpc.handler.Handler([RejectingService()])
        token = temporalio.nexus._token.OperationToken(
            type=temporalio.nexus._token.OperationTokenType(kind),
            namespace="default",
            workflow_id="workflow",
            activity_id="activity",
            run_id="run",
            update_id="update",
        ).encode()
        if kind == 3:
            # Workflow Update backing needs an application-specific cancel handler.
            with pytest.raises(nexusrpc.HandlerError) as error:
                await handler.cancel_operation(ctx, token)
            assert error.value.type == nexusrpc.HandlerErrorType.NOT_IMPLEMENTED
        else:
            await handler.cancel_operation(ctx, token)
            if kind == 1:
                client.get_workflow_handle.assert_called_once_with("workflow")
                client.get_workflow_handle.return_value.cancel.assert_awaited_once()
            else:
                client.get_activity_handle.assert_called_once_with(
                    "activity", run_id="run"
                )
                client.get_activity_handle.return_value.cancel.assert_awaited_once()
        calls = client.mock_calls.copy()
        with pytest.raises(nexusrpc.HandlerError) as error:
            await handler.cancel_operation(ctx, "invalid")
        assert error.value.type == nexusrpc.HandlerErrorType.INTERNAL
        assert client.mock_calls == calls
    finally:
        temporalio.nexus._operation_context._temporal_cancel_operation_context.reset(
            context_token
        )


@nexusrpc.handler.service_handler(name="dynamic-tools")
class DynamicTools:
    def __init__(self, tools: set[str]) -> None:
        self.tools = tools

    @nexusrpc.handler.sync_operation
    async def list_tools(
        self, ctx: nexusrpc.handler.StartOperationContext, input: None
    ) -> list[str]:
        return sorted(self.tools)

    @temporalio.nexus.temporal_operation(dynamic=True)
    async def forward(
        self,
        ctx: temporalio.nexus.TemporalStartOperationContext,
        client: temporalio.nexus.TemporalNexusClient,
        input: nexusrpc.LazyValue,
    ) -> temporalio.nexus.TemporalOperationResult[Any]:
        if ctx.operation not in self.tools:
            raise nexusrpc.HandlerError(
                "Unknown tool", type=nexusrpc.HandlerErrorType.NOT_FOUND
            )
        value = await input.consume(as_type=str)
        if ctx.operation == "echo_sync":
            return temporalio.nexus.TemporalOperationResult.sync(f"sync:{value}")
        if ctx.operation == "echo_workflow":
            return await client.start_workflow(
                EchoWorkflow.run, value, id=str(uuid.uuid4())
            )
        if ctx.operation == "wait_workflow":
            return await client.start_workflow(WaitingWorkflow.run, id=value)
        if ctx.operation == "wait_activity":
            return await client.start_activity(
                wait_for_cancel_activity,
                id=value,
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=1),
            )
        raise nexusrpc.HandlerError(
            "Unknown tool", type=nexusrpc.HandlerErrorType.NOT_FOUND
        )


ALL_TOOLS = {"echo_sync", "echo_workflow", "wait_workflow", "wait_activity"}


@dataclass
class CallerInput:
    endpoint: str
    operation: str
    value: str


@workflow.defn
class DynamicCallerWorkflow:
    @workflow.run
    async def run(self, input: CallerInput) -> Any:
        client = workflow.create_nexus_client(
            service="dynamic-tools", endpoint=input.endpoint
        )
        if input.operation == "list_tools":
            return await client.execute_operation("list_tools", None)
        return await client.execute_operation(input.operation, input.value)


@workflow.defn
class DynamicCancelCallerWorkflow:
    def __init__(self) -> None:
        self.started = False

    @workflow.run
    async def run(self, input: CallerInput) -> None:
        client = workflow.create_nexus_client(
            service="dynamic-tools", endpoint=input.endpoint
        )
        handle: workflow.NexusOperationHandle[Any] = await client.start_operation(
            input.operation, input.value
        )
        self.started = True
        await handle

    @workflow.update
    async def wait_started(self) -> None:
        await workflow.wait_condition(lambda: self.started)


class RecordingInterceptor(Interceptor):
    def __init__(self) -> None:
        self.starts: list[tuple[str, type]] = []
        self.cancels: list[str] = []

    def intercept_nexus_operation(
        self, next: NexusOperationInboundInterceptor
    ) -> NexusOperationInboundInterceptor:
        return _RecordingInbound(next, self)


class _RecordingInbound(NexusOperationInboundInterceptor):
    def __init__(
        self, next: NexusOperationInboundInterceptor, root: RecordingInterceptor
    ) -> None:
        super().__init__(next)
        self._root = root

    async def execute_nexus_operation_start(
        self, input: ExecuteNexusOperationStartInput
    ) -> (
        nexusrpc.handler.StartOperationResultSync[Any]
        | nexusrpc.handler.StartOperationResultAsync
    ):
        self._root.starts.append((input.ctx.operation, type(input.input)))
        return await self.next.execute_nexus_operation_start(input)

    async def execute_nexus_operation_cancel(
        self, input: ExecuteNexusOperationCancelInput
    ) -> None:
        self._root.cancels.append(input.ctx.operation)
        return await self.next.execute_nexus_operation_cancel(input)


async def _setup_endpoint(env: WorkflowEnvironment) -> tuple[str, str]:
    if env.supports_time_skipping_v1:
        pytest.skip("Nexus requires a local server")
    task_queue = str(uuid.uuid4())
    endpoint = make_nexus_endpoint_name(task_queue)
    await env.create_nexus_endpoint(endpoint, task_queue)
    return task_queue, endpoint


def _worker(
    client: Client,
    task_queue: str,
    tools: DynamicTools,
    interceptors: list[Interceptor] | None = None,
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        nexus_service_handlers=[tools],
        workflows=[
            WaitingWorkflow,
            EchoWorkflow,
            DynamicCallerWorkflow,
            DynamicCancelCallerWorkflow,
        ],
        activities=[wait_for_cancel_activity],
        interceptors=interceptors or [],
    )


@pytest.mark.requires_local_server
@pytest.mark.parametrize(
    ["operation", "expected"],
    [
        ("list_tools", sorted(ALL_TOOLS)),
        ("echo_sync", "sync:hello"),
        ("echo_workflow", "hello"),
    ],
)
async def test_workflow_caller_routes_named_and_dynamic_operations(
    client: Client, env: WorkflowEnvironment, operation: str, expected: Any
):
    task_queue, endpoint = await _setup_endpoint(env)
    async with _worker(client, task_queue, DynamicTools(set(ALL_TOOLS))):
        result = await client.execute_workflow(
            DynamicCallerWorkflow.run,
            CallerInput(endpoint=endpoint, operation=operation, value="hello"),
            id=str(uuid.uuid4()),
            task_queue=task_queue,
        )
        assert result == expected


@pytest.mark.requires_local_server
async def test_workflow_caller_gets_not_found_from_dynamic_handler(
    client: Client, env: WorkflowEnvironment
):
    task_queue, endpoint = await _setup_endpoint(env)
    async with _worker(client, task_queue, DynamicTools(set(ALL_TOOLS))):
        with pytest.raises(WorkflowFailureError) as err:
            await client.execute_workflow(
                DynamicCallerWorkflow.run,
                CallerInput(endpoint=endpoint, operation="missing", value="x"),
                id=str(uuid.uuid4()),
                task_queue=task_queue,
            )
        assert isinstance(err.value.__cause__, NexusOperationError)
        handler_error = err.value.__cause__.__cause__
        assert isinstance(handler_error, nexusrpc.HandlerError)
        assert handler_error.type == nexusrpc.HandlerErrorType.NOT_FOUND
        assert not handler_error.retryable


@pytest.mark.requires_local_server
async def test_workflow_caller_cancels_dynamic_workflow_operation(
    client: Client, env: WorkflowEnvironment
):
    task_queue, endpoint = await _setup_endpoint(env)
    interceptor = RecordingInterceptor()
    backing_id = str(uuid.uuid4())
    async with _worker(client, task_queue, DynamicTools(set(ALL_TOOLS)), [interceptor]):
        caller = await client.start_workflow(
            DynamicCancelCallerWorkflow.run,
            CallerInput(endpoint=endpoint, operation="wait_workflow", value=backing_id),
            id=str(uuid.uuid4()),
            task_queue=task_queue,
        )
        await caller.execute_update(DynamicCancelCallerWorkflow.wait_started)
        await caller.cancel()
        with pytest.raises(WorkflowFailureError):
            await client.get_workflow_handle(backing_id).result()
    assert interceptor.starts == [("wait_workflow", nexusrpc.LazyValue)]
    assert interceptor.cancels == ["wait_workflow"]


@pytest.mark.requires_local_server
async def test_standalone_caller_cancels_dynamic_activity_operation(
    client: Client, env: WorkflowEnvironment
):
    task_queue, endpoint = await _setup_endpoint(env)
    interceptor = RecordingInterceptor()
    activity_id = str(uuid.uuid4())
    async with _worker(client, task_queue, DynamicTools(set(ALL_TOOLS)), [interceptor]):
        nexus_client: NexusClient[Any] = client.create_nexus_client(
            "dynamic-tools", endpoint
        )
        handle: NexusOperationHandle[Any] = await nexus_client.start_operation(
            "wait_activity", activity_id, id=str(uuid.uuid4())
        )
        await handle.cancel()
        activity_handle = client.get_activity_handle(activity_id)

        async def cancelled():
            assert (
                await handle.describe()
            ).status is NexusOperationExecutionStatus.CANCELED
            assert (
                await activity_handle.describe()
            ).status is ActivityExecutionStatus.CANCELED

        await assert_eventually(cancelled)
    assert interceptor.starts == [("wait_activity", nexusrpc.LazyValue)]
    assert interceptor.cancels == ["wait_activity"]


@pytest.mark.requires_local_server
async def test_cancel_by_token_after_tool_removal_and_restart(
    client: Client, env: WorkflowEnvironment
):
    task_queue, endpoint = await _setup_endpoint(env)
    tools = DynamicTools({"wait_workflow"})
    nexus_client: NexusClient[Any] = client.create_nexus_client(
        "dynamic-tools", endpoint
    )
    workflow_id = str(uuid.uuid4())
    async with _worker(client, task_queue, tools):
        pending: NexusOperationHandle[Any] = await nexus_client.start_operation(
            "wait_workflow", workflow_id, id=workflow_id
        )

        async def started():
            assert (await pending.describe()).operation_token

        await assert_eventually(started)
        tools.tools.clear()
    # The new Worker has no tools. The dynamic handler cancels from the token alone.
    async with _worker(client, task_queue, DynamicTools(set())):
        await pending.cancel()
        with pytest.raises(WorkflowFailureError):
            await client.get_workflow_handle(workflow_id).result()
