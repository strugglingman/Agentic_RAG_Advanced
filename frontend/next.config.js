/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Produce a self-contained build for Docker (no node_modules needed at runtime)
  output: "standalone",
};
module.exports = nextConfig;