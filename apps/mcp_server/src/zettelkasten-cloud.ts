import { KnowledgeBase } from "./lib/knowledge-base.js";
import { z } from "zod";
import express, { Express, Request, Response } from "express";
import cors from "cors";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

console.log("Starting Zettelkasten server initialization...");

const knowledgeBase = new KnowledgeBase();
console.log("KnowledgeBase initialized");

const server = new McpServer({
  name: "zettelkasten",
  version: "1.0.0",
  capabilities: {
    resources: {},
    tools: {},
  },
});
console.log("McpServer initialized with name: zettelkasten, version: 1.0.0");

server.tool(
  "get_msg_context",
  { userId: z.string().describe("User Id"), query: z.string() },
  async ({ userId, query }) => {
    console.log(
      `Tool get_msg_context called with userId: ${userId}, query: ${query}`,
    );
    try {
      const result = await knowledgeBase.getMsgContext(userId, query);
      console.log(
        `Retrieved message context for user ${userId}, found ${result.messages?.length || 0} messages`,
      );
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result.messages),
          },
        ],
      };
    } catch (error) {
      console.error("Error in get_msg_context tool:", error);
      throw error;
    }
  },
);
console.log("Registered get_msg_context tool");

server.tool(
  "store_message",
  {
    userId: z.string().describe("User Id"),
    messageContent: z.string().describe("Content of the message to store"),
    role: z
      .string()
      .optional()
      .describe("Role of the message (default: 'user')"),
    metadata: z
      .record(z.any())
      .optional()
      .describe("Additional metadata to store with the message"),
  },
  async ({ userId, messageContent, role = "user", metadata = {} }) => {
    const messageId = await knowledgeBase.storeMessage(
      userId,
      transport.sessionId,
      messageContent,
      role,
      metadata,
    );

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ success: true, messageId }),
        },
      ],
    };
  },
);

const app: Express = express();
console.log("Express app created");

app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "OPTIONS"],
    credentials: false,
  }),
);
console.log("CORS middleware configured");

app.get("/", (req: Request, res: Response) => {
  console.log("Received request to root endpoint");
  res.json({
    name: "Zettelkasten",
    version: "1.0.0",
    status: "running",
    endpoints: {
      "/": "Server information (this response)",
      "/sse": "Server-Sent Events endpoint for MCP connection",
      "/messages": "POST endpoint for MCP messages",
    },
    tools: [
      { name: "add", description: "Add two numbers together" },
      { name: "search", description: "Search the web using Brave Search API" },
      {
        name: "get_msg_context",
        description: "Get message context based on a query",
      },
      {
        name: "store_message",
        description: "Store a message in the knowledge base",
      },
    ],
  });
  console.log("Responded to root endpoint");
});

let transport: SSEServerTransport;

app.get("/sse", async (req: Request, res: Response) => {
  console.log("Received request to SSE endpoint");
  transport = new SSEServerTransport("/messages", res);
  console.log("SSE transport created with path: /messages");
  try {
    await server.connect(transport);
    console.log("Server successfully connected to SSE transport");
  } catch (error) {
    console.error("Error connecting server to SSE transport:", error);
    res.status(500).end();
  }
});

app.post("/messages", async (req: Request, res: Response) => {
  console.log("Received POST request to /messages");
  try {
    await transport.handlePostMessage(req, res);
    console.log("Successfully handled POST message");
  } catch (error) {
    console.error("Error handling POST message:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`MCP SSE Server running on port ${PORT}`);
  console.log(`Server endpoints:`);
  console.log(`  - GET /: Server information`);
  console.log(`  - GET /sse: SSE endpoint for MCP connection`);
  console.log(`  - POST /messages: Endpoint for MCP messages`);
});
