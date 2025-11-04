export default function ShimmerBubble() {
  return (
    <>
      <div className="h-10 w-64 rounded-2xl border border-slate-200 bg-gradient-to-r from-slate-100 via-slate-200 to-slate-100 bg-[length:200%_100%] animate-[shimmer_1.2s_linear_infinite]" />
      <style jsx global>{`
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </>
  );
}
