from __future__ import annotations

import json
import re

from .smoke import IMPORT_NAMES as BASE_IMPORT_NAMES
from .smoke import PAIR_TESTS as BASE_PAIR_TESTS


IMPORT_NAMES = {
    **BASE_IMPORT_NAMES,
    "grpcio-tools": "grpc_tools",
    "pydantic-core": "pydantic_core",
}


PAIR_TESTS: dict[frozenset[str], str] = {
    **BASE_PAIR_TESTS,
    frozenset(("fastapi", "starlette")): """
from fastapi import FastAPI
from starlette.testclient import TestClient
app = FastAPI()
@app.get('/deplab')
def status():
    return {'status': 'ok'}
response = TestClient(app).get('/deplab')
assert response.status_code == 200
assert response.json() == {'status': 'ok'}
""",
    frozenset(("fastapi", "pydantic")): """
from fastapi import FastAPI
from pydantic import BaseModel
class Item(BaseModel):
    value: int
app = FastAPI()
@app.post('/items')
def create(item: Item):
    return item
schema = app.openapi()
assert 'Item' in schema['components']['schemas']
""",
    frozenset(("starlette", "anyio")): """
import anyio
from starlette.concurrency import run_in_threadpool
async def check():
    return await run_in_threadpool(lambda: 42)
assert anyio.run(check) == 42
""",
    frozenset(("pydantic", "pydantic-core")): """
from pydantic import BaseModel
import pydantic_core
class Item(BaseModel):
    value: int
assert Item(value='7').value == 7
assert pydantic_core.__version__
""",
    frozenset(("httpx", "anyio")): """
import anyio
import httpx
async def check():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={'ok': True}))
    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get('https://deplab.test')
    return response.json()
assert anyio.run(check) == {'ok': True}
""",
    frozenset(("httpx", "sniffio")): """
import anyio
import httpx
import sniffio
async def check():
    assert sniffio.current_async_library() == 'asyncio'
    transport = httpx.MockTransport(lambda request: httpx.Response(204))
    async with httpx.AsyncClient(transport=transport) as client:
        return (await client.get('https://deplab.test')).status_code
assert anyio.run(check) == 204
""",
    frozenset(("numpy", "pyarrow")): """
import numpy as np
import pyarrow as pa
array = pa.array(np.array([1, 2, 3], dtype=np.int64))
assert array.to_pylist() == [1, 2, 3]
""",
    frozenset(("pandas", "pyarrow")): """
import pandas as pd
import pyarrow as pa
frame = pd.DataFrame({'value': [1, 2, 3]})
table = pa.Table.from_pandas(frame, preserve_index=False)
assert table.column('value').to_pylist() == [1, 2, 3]
""",
    frozenset(("numpy", "polars")): """
import numpy as np
import polars as pl
series = pl.Series('value', np.array([1, 2, 3], dtype=np.int64))
assert series.to_list() == [1, 2, 3]
""",
    frozenset(("pandas", "polars")): """
import pandas as pd
import polars as pl
frame = pl.from_pandas(pd.DataFrame({'value': [1, 2, 3]}))
assert frame['value'].to_list() == [1, 2, 3]
""",
    frozenset(("numpy", "opencv-python")): """
import cv2
import numpy as np
image = np.zeros((4, 4, 3), dtype=np.uint8)
assert cv2.mean(image)[:3] == (0.0, 0.0, 0.0)
""",
    frozenset(("numpy", "dask")): """
import dask.array as da
import numpy as np
array = da.from_array(np.arange(6), chunks=3)
assert array.sum().compute() == 15
""",
    frozenset(("pandas", "dask")): """
import dask
import pandas as pd
frame = pd.DataFrame({'value': [1, 2, 3]})
result = dask.delayed(lambda value: int(value['value'].sum()))(frame).compute()
assert result == 6
""",
    frozenset(("dask", "distributed")): """
import dask
from distributed import Client
client = Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None)
try:
    assert client.submit(lambda: dask.__version__).result() == dask.__version__
finally:
    client.close()
""",
    frozenset(("pandas", "seaborn")): """
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
frame = pd.DataFrame({'x': [1, 2], 'y': [2, 4]})
axis = sns.lineplot(data=frame, x='x', y='y')
assert len(axis.lines) >= 1
""",
    frozenset(("matplotlib", "seaborn")): """
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
figure, axis = plt.subplots()
axis.plot([0, 1], [0, 1])
figure.canvas.draw()
assert sns.color_palette()
plt.close(figure)
""",
    frozenset(("typer", "click")): """
import click
import typer
from typer.testing import CliRunner
app = typer.Typer()
@app.command()
def hello():
    typer.echo('ok')
result = CliRunner().invoke(app, [])
assert result.exit_code == 0
assert 'ok' in result.stdout
assert click.Command
""",
    frozenset(("typer", "rich")): """
import typer
from rich.console import Console
from typer import rich_utils
assert typer.Typer()
assert rich_utils.Console is Console
""",
    frozenset(("boto3", "botocore")): """
import boto3
import botocore
session = boto3.session.Session(
    aws_access_key_id='deplab',
    aws_secret_access_key='deplab',
    region_name='us-east-1',
)
client = session.client('s3', endpoint_url='https://deplab.invalid')
assert client.meta.service_model.service_name == 's3'
assert botocore.__version__
""",
    frozenset(("boto3", "s3transfer")): """
from boto3.s3.transfer import TransferConfig
from s3transfer.manager import TransferConfig as CoreTransferConfig
config = TransferConfig()
assert isinstance(config, CoreTransferConfig)
""",
    frozenset(("celery", "kombu")): """
from celery import Celery
from kombu import Connection
app = Celery('deplab', broker='memory://')
connection = app.connection_for_write()
assert isinstance(connection, Connection)
assert connection.transport.driver_type == 'memory'
connection.close()
""",
    frozenset(("celery", "billiard")): """
from billiard.pool import Pool
from celery.concurrency.prefork import TaskPool
assert Pool is not None
assert TaskPool is not None
""",
    frozenset(("sqlalchemy", "greenlet")): """
import asyncio
from greenlet import getcurrent
from sqlalchemy.util.concurrency import greenlet_spawn
async def check():
    parent = getcurrent()
    child = await greenlet_spawn(getcurrent)
    return parent is not child
assert asyncio.run(check())
""",
    frozenset(("pytest", "pluggy")): """
import pluggy
import pytest
hookspec = pluggy.HookspecMarker('deplab')
hookimpl = pluggy.HookimplMarker('deplab')
class Spec:
    @hookspec
    def value(self):
        pass
class Plugin:
    @hookimpl
    def value(self):
        return 7
manager = pluggy.PluginManager('deplab')
manager.add_hookspecs(Spec)
manager.register(Plugin())
assert manager.hook.value() == [7]
assert pytest.hookimpl
""",
    frozenset(("grpcio-tools", "protobuf")): """
from google.protobuf import descriptor_pb2
from grpc_tools import protoc
descriptor = descriptor_pb2.FileDescriptorProto(name='deplab.proto')
assert descriptor.name == 'deplab.proto'
assert callable(protoc.main)
""",
    frozenset(("aiohttp", "yarl")): """
from aiohttp.client_reqrep import URL as AiohttpURL
from yarl import URL
assert AiohttpURL is URL
assert str(AiohttpURL('https://deplab.test/path')) == 'https://deplab.test/path'
""",
    frozenset(("yarl", "multidict")): """
from multidict import MultiDict
from yarl import URL
url = URL('https://deplab.test').with_query(MultiDict([('value', '1'), ('value', '2')]))
assert url.query.getall('value') == ['1', '2']
""",
}


def import_name(package: str) -> str:
    normalized = re.sub(r"[-_.]+", "-", package).lower()
    return IMPORT_NAMES.get(normalized, normalized.replace("-", "_"))


def build_smoke_script(package_a: str, package_b: str) -> str:
    normalized = [re.sub(r"[-_.]+", "-", value).lower() for value in (package_a, package_b)]
    imports = "\n".join(f"import {import_name(package)}" for package in normalized)
    pair_test = PAIR_TESTS.get(frozenset(normalized))
    strength = "interoperability" if pair_test else "imports"
    body = pair_test or ""
    return f"""# Generated by DepLab v3; executed inside an isolated environment.
import json
{imports}
print(json.dumps({{"deplab_stage": "imports_passed"}}))
{body}
print(json.dumps({{"deplab_smoke": "pass", "strength": {json.dumps(strength)}}}))
"""
