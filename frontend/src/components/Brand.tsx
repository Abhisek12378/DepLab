import { Boxes } from "lucide-react";

export function Brand() {
  return (
    <div className="brand" aria-label="DepLab">
      <span className="brand-mark"><Boxes size={19} strokeWidth={2.2} /></span>
      <span className="brand-name">DepLab</span>
      <span className="brand-badge">BETA</span>
    </div>
  );
}
