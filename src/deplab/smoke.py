from __future__ import annotations

import json
import re


IMPORT_NAMES = {
    "beautifulsoup4": "bs4",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "pyopenssl": "OpenSSL",
}


PAIR_TESTS: dict[frozenset[str], str] = {
    frozenset(("numpy", "pandas")): """
import numpy as np
import pandas as pd
frame = pd.DataFrame(np.arange(6).reshape(2, 3))
assert frame.sum().tolist() == [3, 5, 7]
""",
    frozenset(("numpy", "scipy")): """
import numpy as np
from scipy import linalg
solution = linalg.solve(np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0]))
assert np.allclose(solution, [2.0, 3.0])
""",
    frozenset(("numpy", "scikit-learn")): """
import numpy as np
from sklearn.preprocessing import StandardScaler
scaled = StandardScaler().fit_transform(np.array([[0.0], [2.0], [4.0]]))
assert np.isclose(float(scaled.mean()), 0.0)
""",
    frozenset(("numpy", "pillow")): """
import numpy as np
from PIL import Image
image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
assert np.asarray(image).shape == (4, 4, 3)
""",
    frozenset(("scipy", "scikit-learn")): """
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
features = csr_matrix(np.array([[0.0], [1.0], [2.0], [3.0]]))
model = LogisticRegression().fit(features, np.array([0, 0, 1, 1]))
assert model.predict(csr_matrix([[2.5]])).tolist() == [1]
""",
    frozenset(("pandas", "scikit-learn")): """
import pandas as pd
from sklearn.preprocessing import StandardScaler
frame = pd.DataFrame({'value': [0.0, 2.0, 4.0]})
scaled = StandardScaler().fit_transform(frame)
assert scaled.shape == (3, 1)
""",
    frozenset(("numpy", "matplotlib")): """
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
figure, axis = plt.subplots()
axis.plot(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]))
figure.canvas.draw()
assert figure.canvas.get_width_height()[0] > 0
plt.close(figure)
""",
    frozenset(("numpy", "numba")): """
import numpy as np
from numba import njit
@njit
def total(values):
    return values.sum()
assert total(np.array([1, 2, 3], dtype=np.int64)) == 6
""",
    frozenset(("numpy", "xarray")): """
import numpy as np
import xarray as xr
array = xr.DataArray(np.arange(6).reshape(2, 3), dims=('row', 'column'))
assert float(array.mean()) == 2.5
""",
    frozenset(("pandas", "xarray")): """
import pandas as pd
import xarray as xr
frame = pd.DataFrame({'value': [1, 2]}, index=pd.Index(['a', 'b'], name='item'))
dataset = xr.Dataset.from_dataframe(frame)
assert dataset['value'].sel(item='b').item() == 2
""",
    frozenset(("matplotlib", "pillow")): """
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
buffer = BytesIO()
figure, axis = plt.subplots()
axis.plot([0, 1], [0, 1])
figure.savefig(buffer, format='png')
plt.close(figure)
buffer.seek(0)
image = Image.open(buffer)
assert image.format == 'PNG'
""",
    frozenset(("requests", "urllib3")): """
import requests
import urllib3
assert isinstance(requests.Session().get_adapter('https://').poolmanager, urllib3.PoolManager)
""",
    frozenset(("flask", "werkzeug")): """
from flask import Flask
app = Flask('deplab_smoke')
@app.get('/deplab')
def deplab_route():
    return {'status': 'ok'}
response = app.test_client().get('/deplab')
assert response.status_code == 200
assert response.get_json() == {'status': 'ok'}
""",
    frozenset(("flask", "jinja2")): """
from flask import Flask
from jinja2 import Environment
app = Flask('deplab_template_smoke')
template = app.jinja_env.from_string('{{ value|upper }}')
assert isinstance(app.jinja_env, Environment)
assert template.render(value='deplab') == 'DEPLAB'
""",
    frozenset(("jinja2", "markupsafe")): """
from jinja2 import Environment
from markupsafe import Markup
template = Environment(autoescape=True).from_string('{{ value }}|{{ trusted }}')
rendered = template.render(value='<unsafe>', trusted=Markup('<strong>safe</strong>'))
assert rendered == '&lt;unsafe&gt;|<strong>safe</strong>'
""",
    frozenset(("httpx", "httpcore")): """
import httpx
import httpcore
transport = httpx.HTTPTransport()
assert isinstance(transport._pool, httpcore.ConnectionPool)
transport.close()
""",
    frozenset(("django", "asgiref")): """
import django
from asgiref.sync import async_to_sync
async def django_version():
    return django.get_version()
assert async_to_sync(django_version)() == django.get_version()
""",
    frozenset(("cryptography", "pyopenssl")): """
from OpenSSL import crypto
from cryptography import x509
key = crypto.PKey()
key.generate_key(crypto.TYPE_RSA, 1024)
certificate = crypto.X509()
certificate.get_subject().CN = 'deplab.test'
certificate.set_serial_number(1)
certificate.gmtime_adj_notBefore(0)
certificate.gmtime_adj_notAfter(60)
certificate.set_issuer(certificate.get_subject())
certificate.set_pubkey(key)
certificate.sign(key, 'sha256')
pem = crypto.dump_certificate(crypto.FILETYPE_PEM, certificate)
loaded = x509.load_pem_x509_certificate(pem)
assert loaded.subject.rfc4514_string() == 'CN=deplab.test'
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
    return f"""# Generated by DepLab; executed inside an isolated environment.
import json
{imports}
print(json.dumps({{"deplab_stage": "imports_passed"}}))
{body}
print(json.dumps({{"deplab_smoke": "pass", "strength": {json.dumps(strength)}}}))
"""
