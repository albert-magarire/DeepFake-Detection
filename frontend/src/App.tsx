import { useEffect, useState } from "react";

function App() {
  const host = (import.meta.env.VITE_HOST || import.meta.env.VITE_LOCAL_HOST || "http://127.0.0.1:8000").replace(/\/$/, "");

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<{
    label: string;
    confidence: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const handleFileChange = (file: File | null) => {
    if (!file) return;

    setFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${host}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Image analysis failed.");
      }
      if (!["Real", "Fake"].includes(data.label) || typeof data.confidence !== "number" ||
          !Number.isFinite(data.confidence) || data.confidence < 0 || data.confidence > 1) {
        throw new Error("The API returned an invalid prediction.");
      }
      const confidence = data.label === "Fake" ? data.confidence : 1 - data.confidence;
      setResult({ label: data.label, confidence });
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to reach the API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center p-6">

      <div className="bg-gray-900 border border-gray-800 shadow-2xl rounded-2xl p-8 w-full max-w-lg">

        <h1 className="text-3xl font-bold text-center mb-2">
          Deepfake Detection
        </h1>

        <p className="text-gray-400 text-center mb-6">
          Upload an image to analyze authenticity
        </p>

        <input
          type="file"
          accept="image/*"
          disabled={loading}
          className="mb-4 w-full text-sm text-gray-400
                     file:mr-4 file:py-2 file:px-4
                     file:rounded-lg file:border-0
                     file:bg-blue-600 file:text-white
                     hover:file:bg-blue-700"
          onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
        />

        {preview && (
          <div className="mb-4">
            <img
              src={preview}
              alt="preview"
              className="rounded-xl w-full max-h-80 object-contain border border-gray-800"
            />
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 transition px-4 py-2 rounded-lg font-semibold"
        >
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>

        {error && <p role="alert" className="mt-4 text-red-400">{error}</p>}

        {result && (
          <div className="mt-6 text-center">

            <h2 className="text-xl font-semibold mb-2">
              Result:{" "}
              <span
                className={
                  result.label === "Fake"
                    ? "text-red-500"
                    : "text-green-400"
                }
              >
                {result.label.toUpperCase()}
              </span>
            </h2>

            <p className="text-gray-300 mb-3">
              Confidence: {(result.confidence * 100).toFixed(2)}%
            </p>

            <div className="w-full bg-gray-800 rounded-full h-3">
              <div
                className={`h-3 rounded-full ${
                  result.label === "Fake"
                    ? "bg-red-500"
                    : "bg-green-400"
                }`}
                style={{
                  width: `${result.confidence * 100}%`,
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
