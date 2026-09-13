import http from "node:http";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const HOST = process.env.FWS_BRIDGE_HOST ?? "127.0.0.1";
const PORT = Number(process.env.FWS_BRIDGE_PORT ?? "8787");
const COMMAND = process.env.FWS_MCP_COMMAND ?? "uvx";
const COMMAND_ARGS = process.env.FWS_MCP_ARGS
  ? process.env.FWS_MCP_ARGS.split(" ").filter(Boolean)
  : ["free-search-mcp"];
const TOKEN = process.env.FWS_BRIDGE_TOKEN ?? "";
const MAX_BODY = 64 * 1024;
const ALLOWED_TOOLS = ["search", "research", "fetch", "read_document"];

const client = new Client({ name: "lfm-mobile-free-search-mcp-bridge", version: "0.3.0" });
const transport = new StdioClientTransport({ command: COMMAND, args: COMMAND_ARGS, stderr: "inherit" });
let connected = false;
let queue = Promise.resolve();

async function ensureConnected() {
  if (!connected) {
    await client.connect(transport);
    connected = true;
  }
}

function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(payload),
    "Cache-Control": "no-store",
  });
  res.end(payload);
}

function authorized(req) {
  if (!TOKEN) return true;
  return req.headers.authorization === `Bearer ${TOKEN}`;
}

async function callTool(name, args) {
  await ensureConnected();
  return client.callTool({ name, arguments: args ?? {} });
}

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    return json(res, 200, { ok: true, connected, authRequired: Boolean(TOKEN), server: "free-search-mcp" });
  }
  if (req.method !== "POST" || req.url !== "/call") return json(res, 404, { error: "not found" });
  if (!authorized(req)) return json(res, 401, { ok: false, error: "unauthorized" });

  let size = 0;
  const chunks = [];
  req.on("data", chunk => {
    size += chunk.length;
    if (size <= MAX_BODY) chunks.push(chunk);
  });
  req.on("end", () => {
    if (size > MAX_BODY) return json(res, 413, { error: "request too large" });
    let body;
    try {
      body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    } catch {
      return json(res, 400, { error: "invalid JSON" });
    }
    const name = typeof body?.tool === "string" ? body.tool : "";
    if (!ALLOWED_TOOLS.includes(name)) return json(res, 400, { error: "tool not allowed" });

    queue = queue.then(async () => {
      try {
        const result = await callTool(name, body.arguments);
        json(res, 200, { ok: true, result });
      } catch (error) {
        connected = false;
        json(res, 502, { ok: false, error: error instanceof Error ? error.message : String(error) });
      }
    }).catch(error => json(res, 500, { ok: false, error: String(error) }));
  });
});

server.listen(PORT, HOST, () => console.log(`free-search-mcp bridge listening on http://${HOST}:${PORT}`));

async function shutdown() {
  server.close();
  try { await transport.close(); } catch {}
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
