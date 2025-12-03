import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 text-center">
      <div className="max-w-3xl">
        <h1 className="text-5xl font-bold mb-4">
          Zen Live
        </h1>
        <p className="text-xl text-gray-600 dark:text-gray-400 mb-8">
          Real-time Speech Translation for Broadcast
        </p>
        <p className="text-lg text-gray-500 dark:text-gray-500 mb-12">
          Low-latency simultaneous translation service for news control rooms.
          Powered by Hanzo AI infrastructure.
        </p>

        <div className="flex gap-4 justify-center flex-wrap">
          <Link
            href="/docs"
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
          >
            Documentation
          </Link>
          <a
            href="https://zen-live.hanzo.ai"
            className="px-6 py-3 border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition"
          >
            Live Demo
          </a>
          <a
            href="https://github.com/zenlm/zen-live"
            className="px-6 py-3 border border-gray-300 dark:border-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition"
          >
            GitHub
          </a>
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-left">
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">⚡ Low Latency</h3>
            <p className="text-gray-600 dark:text-gray-400">
              ~3 second round-trip translation with real-time audio streaming
            </p>
          </div>
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">🌍 18 Languages</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Support for Spanish, English, Chinese, and 15 more languages
            </p>
          </div>
          <div className="p-6 border border-gray-200 dark:border-gray-800 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">🎙️ Multiple Voices</h3>
            <p className="text-gray-600 dark:text-gray-400">
              8 natural-sounding voices for translated audio output
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
