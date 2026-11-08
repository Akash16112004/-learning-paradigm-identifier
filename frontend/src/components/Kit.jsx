import { PARADIGM } from "../lib/api";

export const Panel = ({ title, kicker, children, className = "", testid, right }) => (
  <section data-testid={testid} className={`panel p-6 rise ${className}`}>
    {(title || kicker) && (
      <div className="mb-5 flex items-start justify-between gap-4">
        <div>
          {kicker && <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-cyan-400/80">{kicker}</div>}
          {title && <h3 className="mt-1 text-lg font-semibold text-slate-100">{title}</h3>}
        </div>
        {right}
      </div>
    )}
    {children}
  </section>
);

export const ParadigmBadge = ({ label, testid }) => (
  <span data-testid={testid} className={`inline-flex items-center rounded border px-2.5 py-1 font-mono text-xs uppercase tracking-wider ${PARADIGM[label]?.badge || PARADIGM.not_ml.badge}`}>
    {label?.replace("_", " ") || "—"}
  </span>
);

export const ProbBars = ({ probs, testid }) => (
  <div data-testid={testid} className="space-y-2.5">
    {Object.entries(probs || {}).map(([k, v]) => (
      <div key={k} className="grid grid-cols-[110px_1fr_56px] items-center gap-3">
        <span className="font-mono text-xs uppercase text-slate-400">{k.replace("_", " ")}</span>
        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full transition-[width] duration-700" style={{ width: `${v * 100}%`, background: PARADIGM[k]?.hex }} />
        </div>
        <span className="text-right font-mono text-xs text-slate-200">{(v * 100).toFixed(1)}%</span>
      </div>
    ))}
  </div>
);

export const Stat = ({ label, value, sub, testid, color = "text-slate-100" }) => (
  <div data-testid={testid} className="panel p-5">
    <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-slate-500">{label}</div>
    <div className={`mt-2 font-display text-3xl font-bold ${color}`}>{value}</div>
    {sub && <div className="mt-1 text-xs text-slate-400">{sub}</div>}
  </div>
);

export const PageHead = ({ unit, title, children }) => (
  <header className="mb-10 rise">
    <div className="font-mono text-xs uppercase tracking-[0.25em] text-cyan-400">{unit}</div>
    <h1 className="mt-3 text-4xl font-extrabold tracking-tight text-slate-50 sm:text-5xl">{title}</h1>
    {children && <p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-400 sm:text-base">{children}</p>}
  </header>
);
