import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import WorkbenchPage from "./pages/WorkbenchPage";
import AuthGate from "./components/AuthGate";

export default function App() {
  const [dataset, setDataset] = useState(null);

  return (
    <AuthGate><div className="min-h-screen bg-graphite-950">
      {dataset ? (
        <WorkbenchPage dataset={dataset} />
      ) : (
        <UploadPage onUploaded={setDataset} />
      )}
    </div></AuthGate>
  );
}
