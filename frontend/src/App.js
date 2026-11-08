import "@/App.css";
import { BrowserRouter, Routes, Route, NavLink, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Toaster } from "./components/ui/sonner";
import { ScanSearch, Database, Cpu, BarChart3, BookOpen, ShieldCheck } from "lucide-react";
import Classifier from "./pages/Classifier";
import Dataset from "./pages/Dataset";
import Training from "./pages/Training";
import Evaluation from "./pages/Evaluation";
import Foundations from "./pages/Foundations";
import Responsible from "./pages/Responsible";

const NAV = [
  ["/", "classifier", "Classify", ScanSearch],
  ["/data", "synthetic-data", "Synthetic data", Database],
  ["/training", "training", "Keras training", Cpu],
  ["/evaluation", "evaluation", "Evaluation", BarChart3],
  ["/foundations", "foundations", "Foundations", BookOpen],
  ["/responsible", "responsible-ai", "Responsible AI", ShieldCheck],
];

const Nav = () => (
  <nav className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/80 backdrop-blur-xl">
    <div className="mx-auto flex max-w-7xl items-center gap-6 overflow-x-auto px-4 py-3 sm:px-6 lg:px-8">
      <div className="flex shrink-0 items-center gap-2" data-testid="brand">
        <div className="grid h-8 w-8 place-items-center rounded bg-cyan-500 font-mono text-sm font-bold text-slate-950">P2</div>
        <span className="hidden font-display text-sm font-semibold text-slate-100 md:block">Paradigm Identifier</span>
      </div>
      <div className="flex gap-1">
        {NAV.map(([to, id, label, Icon]) => (
          <NavLink key={id} to={to} end data-testid={`nav-item-${id}`}
            className={({ isActive }) => `flex shrink-0 items-center gap-2 rounded px-3 py-2 text-sm transition-colors ${isActive ? "bg-cyan-500/10 text-cyan-300" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}>
            <Icon size={15} /> {label}
          </NavLink>
        ))}
      </div>
    </div>
  </nav>
);

const ScrollTop = () => {
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return null;
};

export default function App() {
  return (
    <BrowserRouter>
      <ScrollTop />
      <Nav />
      <main className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Classifier />} />
          <Route path="/data" element={<Dataset />} />
          <Route path="/training" element={<Training />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/foundations" element={<Foundations />} />
          <Route path="/responsible" element={<Responsible />} />
        </Routes>
      </main>
      <footer className="border-t border-slate-800 py-8 text-center font-mono text-xs text-slate-500">
        P2 · Learning-Paradigm Identifier · Generative AI mini-project (U1–U5)
      </footer>
      <Toaster theme="dark" position="bottom-right" />
    </BrowserRouter>
  );
}
