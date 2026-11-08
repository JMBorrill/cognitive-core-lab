import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Pages read web/public/data/*.json from disk at request time; make sure the files
  // travel with the server bundle if this is ever deployed as a standalone/serverless build.
  outputFileTracingIncludes: {
    "/**": ["./public/data/**"],
  },
};

export default nextConfig;
