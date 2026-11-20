import { createRoot } from "react-dom/client";
import WorkbenchPage from "../src/pages/WorkbenchPage.jsx";

export function mount(container, dataset) {
  const root = createRoot(container);
  root.render(<WorkbenchPage dataset={dataset} />);
  return () => root.unmount();
}
