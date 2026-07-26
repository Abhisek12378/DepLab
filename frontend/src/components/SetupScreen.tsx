import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { ArrowRight, FileCode2, LockKeyhole, Sparkles, Upload } from "lucide-react";
import { SUPPORTED_PYTHON_VERSIONS, type CreateConversationInput } from "../types";
import { Brand } from "./Brand";

const SAMPLE = "numpy==1.26.4\npandas==2.1.4\nscipy==1.11.4\nrequests==2.31.0";
const MAX_FILE_BYTES = 100_000;

interface SetupScreenProps {
  busy: boolean;
  error: string | null;
  onStart: (input: CreateConversationInput) => Promise<void>;
}

export function SetupScreen({ busy, error, onStart }: SetupScreenProps) {
  const [requirements, setRequirements] = useState(SAMPLE);
  const [pythonVersion, setPythonVersion] = useState<CreateConversationInput["python_version"]>("3.11");
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      setFileError("requirements.txt must be smaller than 100 KB.");
      return;
    }
    setFileError(null);
    setRequirements(await file.text());
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!requirements.trim()) {
      setFileError("Add at least one package requirement.");
      return;
    }
    await onStart({
      requirements_text: requirements.trim(),
      python_version: pythonVersion,
      platform: "linux-x86_64",
    });
  }

  return (
    <main className="setup-page">
      <header className="setup-nav">
        <Brand />
        <div className="privacy-note"><LockKeyhole size={14} /> Session memory expires automatically</div>
      </header>

      <section className="setup-grid">
        <div className="hero-copy">
          <div className="eyebrow"><Sparkles size={15} /> Dependency intelligence, with evidence</div>
          <h1>Upgrade Python packages with <span>fewer surprises.</span></h1>
          <p className="hero-lead">
            Ask DepLab what may break, why it matters, and which complete environment is the safest next move.
          </p>
          <div className="proof-row" aria-label="DepLab evidence types">
            <div><strong>21,490</strong><span>training combinations</span></div>
            <div><strong>2 signals</strong><span>constraints + ML risk</span></div>
            <div><strong>Clear</strong><span>fact vs prediction</span></div>
          </div>
        </div>

        <form className="setup-card" onSubmit={handleSubmit}>
          <div className="setup-card-header">
            <div className="step-number">01</div>
            <div><h2>Describe your environment</h2><p>We use this as the source of truth for the conversation.</p></div>
          </div>

          <div className="field-row">
            <label className="field compact-field">
              <span>Python version</span>
              <select value={pythonVersion} onChange={(event) => setPythonVersion(event.target.value as CreateConversationInput["python_version"])}>
                {SUPPORTED_PYTHON_VERSIONS.map((version) => (
                  <option key={version} value={version}>Python {version}</option>
                ))}
              </select>
            </label>
            <div className="field compact-field"><span>Target platform</span><div className="static-input">Linux · x86_64</div></div>
          </div>

          <label className="field requirements-field">
            <span>requirements.txt</span>
            <textarea value={requirements} onChange={(event) => setRequirements(event.target.value)} spellCheck={false} aria-describedby="requirements-help" />
          </label>
          <div className="requirements-meta" id="requirements-help">
            <button className="upload-button" type="button" onClick={() => fileInput.current?.click()}>
              <Upload size={15} /> Upload file
            </button>
            <span><FileCode2 size={14} /> Exact pins give the strongest answer</span>
            <input ref={fileInput} className="visually-hidden" type="file" accept=".txt,text/plain" onChange={handleFile} />
          </div>

          {(error || fileError) && <div className="form-error" role="alert">{fileError || error}</div>}
          <button className="primary-button" type="submit" disabled={busy}>
            {busy ? "Preparing your workspace…" : "Start dependency analysis"}
            {!busy && <ArrowRight size={17} />}
          </button>
          <p className="safety-copy">No packages are installed. uv checks resolution; DepLab predicts post-install risk.</p>
        </form>
      </section>
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
    </main>
  );
}
