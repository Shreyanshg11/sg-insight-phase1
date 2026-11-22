import { createRoot } from "react-dom/client";
import ErrorBoundary from "../src/components/ErrorBoundary.jsx";
import WorkbenchPage from "../src/pages/WorkbenchPage.jsx";

export function mount(container, dataset) {
  const root = createRoot(container);
  root.render(<WorkbenchPage dataset={dataset} />);
  return () => root.unmount();
}

// eslint-disable-next-line react/only-export-components
function Boom() { throw new Error("simulated render crash"); }

export function mountCrash(container) {
  const root = createRoot(container);
  root.render(<ErrorBoundary><Boom /></ErrorBoundary>);
  return () => root.unmount();
}
