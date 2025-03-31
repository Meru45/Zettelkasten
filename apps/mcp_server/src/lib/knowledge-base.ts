import { config } from "dotenv";
import * as neo4j from "neo4j-driver";
import { Index, Pinecone, RecordMetadata } from "@pinecone-database/pinecone";
import { OpenAI } from "openai";
import { v4 as uuidv4 } from "uuid";

config();

// Configuration
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || "";
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || "chat-context";
const NEO4J_URI = process.env.NEO4J_URI || "";
const NEO4J_USER = process.env.NEO4J_USER || "";
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-ada-002";
const VECTOR_DIMENSION = process.env.VECTOR_DIMENSION || 1536;
const CONTEXT_CACHE_SIZE = parseInt(process.env.CONTEXT_CACHE_SIZE || "100");
const SESSION_CACHE_SIZE = parseInt(process.env.SESSION_CACHE_SIZE || "50");

process.stdout.write(
  `[CONFIG] Initializing with: PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME}, VECTOR_DIMENSION=${VECTOR_DIMENSION}, CONTEXT_CACHE_SIZE=${CONTEXT_CACHE_SIZE}, SESSION_CACHE_SIZE=${SESSION_CACHE_SIZE}\n`,
);

// Types
export interface MessageMetadata {
  [key: string]: any;
}

export interface ContextResult {
  messages: Array<{
    content: string;
    role: string;
    session_id: string;
    timestamp: number;
    metadata?: MessageMetadata;
  }>;
  relatedSessions: string[];
}

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  lastAccessed: number;
  embedding: number[];
  key: string;
}

class SemanticCache<T> {
  private cache: Map<string, CacheEntry<T>>;
  private maxSize: number;
  private openai: OpenAI;
  private cacheHits: number = 0;
  private cacheMisses: number = 0;

  constructor(maxSize: number, openai: OpenAI) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.openai = openai;
    process.stdout.write(
      `[SEMANTIC_CACHE] Initialized with max size: ${maxSize}\n`,
    );
  }

  public async get(key: string): Promise<T | null> {
    process.stdout.write(`[SEMANTIC_CACHE] Attempting to get: ${key}\n`);
    const entry = this.cache.get(key);
    if (entry) {
      entry.lastAccessed = Date.now();
      this.cacheHits++;
      process.stdout.write(
        `[SEMANTIC_CACHE] Cache HIT for key: ${key}, total hits: ${this.cacheHits}\n`,
      );
      return entry.data;
    }
    this.cacheMisses++;
    process.stdout.write(
      `[SEMANTIC_CACHE] Cache MISS for key: ${key}, total misses: ${this.cacheMisses}\n`,
    );
    return null;
  }

  public async set(key: string, data: T, queryText: string): Promise<void> {
    process.stdout.write(
      `[SEMANTIC_CACHE] Setting cache entry for key: ${key}, query length: ${queryText.length}\n`,
    );
    const embedding = await this.generateEmbedding(queryText);

    if (this.cache.size >= this.maxSize) {
      process.stdout.write(
        `[SEMANTIC_CACHE] Cache full (size: ${this.cache.size}), evicting entry\n`,
      );
      await this.evictLeastSimilar(embedding);
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      lastAccessed: Date.now(),
      embedding,
      key,
    });
    process.stdout.write(
      `[SEMANTIC_CACHE] Successfully set cache entry, new size: ${this.cache.size}\n`,
    );
  }

  public getStats(): {
    hits: number;
    misses: number;
    size: number;
    hitRate: number;
  } {
    const total = this.cacheHits + this.cacheMisses;
    const hitRate = total === 0 ? 0 : this.cacheHits / total;
    process.stdout.write(
      `[SEMANTIC_CACHE] Stats - Hits: ${this.cacheHits}, Misses: ${this.cacheMisses}, Size: ${this.cache.size}, Hit Rate: ${hitRate.toFixed(2)}\n`,
    );
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses,
      size: this.cache.size,
      hitRate,
    };
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    process.stdout.write(
      `[SEMANTIC_CACHE] Generating embedding for text of length: ${text.length}\n`,
    );
    try {
      const response = await this.openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: text,
      });
      process.stdout.write(
        `[SEMANTIC_CACHE] Successfully generated embedding with dimension: ${response.data[0].embedding.length}\n`,
      );
      return response.data[0].embedding;
    } catch (error) {
      process.stderr.write(
        `[SEMANTIC_CACHE] Error generating embedding for cache: ${error}\n`,
      );
      process.stdout.write(
        `[SEMANTIC_CACHE] Returning zero vector with dimension: ${VECTOR_DIMENSION}\n`,
      );
      return new Array(parseInt(VECTOR_DIMENSION.toString())).fill(0);
    }
  }

  private async evictLeastSimilar(newEmbedding: number[]): Promise<void> {
    process.stdout.write(
      `[SEMANTIC_CACHE] Starting eviction process for cache with size: ${this.cache.size}\n`,
    );
    if (this.cache.size === 0) {
      process.stdout.write(
        `[SEMANTIC_CACHE] Cache is empty, no eviction needed\n`,
      );
      return;
    }

    const entries = Array.from(this.cache.values());
    let leastSimilarKey = entries[0].key;
    let lowestSimilarity = 1.0;

    for (const entry of entries) {
      const similarity = this.cosineSimilarity(newEmbedding, entry.embedding);

      const recencyScore =
        (Date.now() - entry.lastAccessed) / (24 * 60 * 60 * 1000); // Normalize to days
      const hybridScore = similarity - recencyScore * 0.2; // Adjust weight as needed

      process.stdout.write(
        `[SEMANTIC_CACHE] Entry ${entry.key}: similarity=${similarity.toFixed(4)}, recencyScore=${recencyScore.toFixed(4)}, hybridScore=${hybridScore.toFixed(4)}\n`,
      );

      if (hybridScore < lowestSimilarity) {
        lowestSimilarity = hybridScore;
        leastSimilarKey = entry.key;
      }
    }

    process.stdout.write(
      `[SEMANTIC_CACHE] Evicting entry with key: ${leastSimilarKey}, score: ${lowestSimilarity.toFixed(4)}\n`,
    );
    this.cache.delete(leastSimilarKey);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    normA = Math.sqrt(normA);
    normB = Math.sqrt(normB);

    if (normA === 0 || normB === 0) {
      return 0;
    }

    return dotProduct / (normA * normB);
  }
}

class HybridLRUMRUCache<T> {
  private cache: Map<
    string,
    { data: T; timestamp: number; accessCount: number; lastAccessed: number }
  >;
  private maxSize: number;
  private cacheHits: number = 0;
  private cacheMisses: number = 0;

  constructor(maxSize: number) {
    this.cache = new Map();
    this.maxSize = maxSize;
    process.stdout.write(
      `[HYBRID_CACHE] Initialized with max size: ${maxSize}\n`,
    );
  }

  public get(key: string): T | null {
    process.stdout.write(`[HYBRID_CACHE] Attempting to get: ${key}\n`);
    const entry = this.cache.get(key);
    if (entry) {
      // Update access count
      entry.accessCount++;
      entry.lastAccessed = Date.now();
      this.cacheHits++;
      process.stdout.write(
        `[HYBRID_CACHE] Cache HIT for key: ${key}, access count: ${entry.accessCount}, total hits: ${this.cacheHits}\n`,
      );
      return entry.data;
    }
    this.cacheMisses++;
    process.stdout.write(
      `[HYBRID_CACHE] Cache MISS for key: ${key}, total misses: ${this.cacheMisses}\n`,
    );
    return null;
  }

  public set(key: string, data: T): void {
    process.stdout.write(
      `[HYBRID_CACHE] Setting cache entry for key: ${key}\n`,
    );
    if (this.cache.size >= this.maxSize) {
      process.stdout.write(
        `[HYBRID_CACHE] Cache full (size: ${this.cache.size}), evicting entry\n`,
      );
      this.evict();
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      accessCount: 1,
      lastAccessed: Date.now(),
    });
    process.stdout.write(
      `[HYBRID_CACHE] Successfully set cache entry, new size: ${this.cache.size}\n`,
    );
  }

  public getStats(): {
    hits: number;
    misses: number;
    size: number;
    hitRate: number;
  } {
    const total = this.cacheHits + this.cacheMisses;
    const hitRate = total === 0 ? 0 : this.cacheHits / total;
    process.stdout.write(
      `[HYBRID_CACHE] Stats - Hits: ${this.cacheHits}, Misses: ${this.cacheMisses}, Size: ${this.cache.size}, Hit Rate: ${hitRate.toFixed(2)}\n`,
    );
    return {
      hits: this.cacheHits,
      misses: this.cacheMisses,
      size: this.cache.size,
      hitRate,
    };
  }

  private evict(): void {
    process.stdout.write(
      `[HYBRID_CACHE] Starting eviction process for cache with size: ${this.cache.size}\n`,
    );
    if (this.cache.size === 0) {
      process.stdout.write(
        `[HYBRID_CACHE] Cache is empty, no eviction needed\n`,
      );
      return;
    }

    const entries = Array.from(this.cache.entries());

    entries.sort((a, b) => {
      const aEntry = a[1];
      const bEntry = b[1];

      const aRecency = Date.now() - aEntry.lastAccessed;
      const bRecency = Date.now() - bEntry.lastAccessed;

      // Normalize recency (newer is better, so inverse)
      const maxRecency = Math.max(aRecency, bRecency);
      const normalizedARecency = 1 - aRecency / maxRecency;
      const normalizedBRecency = 1 - bRecency / maxRecency;

      // Normalize frequency (more accesses is better)
      const maxFreq = Math.max(aEntry.accessCount, bEntry.accessCount);
      const normalizedAFreq = aEntry.accessCount / maxFreq;
      const normalizedBFreq = bEntry.accessCount / maxFreq;

      const aScore = normalizedAFreq * 0.6 + normalizedARecency * 0.4;
      const bScore = normalizedBFreq * 0.6 + normalizedBRecency * 0.4;

      process.stdout.write(
        `[HYBRID_CACHE] Entry ${a[0]}: accessCount=${aEntry.accessCount}, recency=${aRecency}ms, score=${aScore.toFixed(4)}\n`,
      );
      process.stdout.write(
        `[HYBRID_CACHE] Entry ${b[0]}: accessCount=${bEntry.accessCount}, recency=${bRecency}ms, score=${bScore.toFixed(4)}\n`,
      );

      return aScore - bScore;
    });

    // Remove lowest hybrid score
    process.stdout.write(
      `[HYBRID_CACHE] Evicting entry with key: ${entries[0][0]}\n`,
    );
    this.cache.delete(entries[0][0]);
  }
}

export class KnowledgeBase {
  private neo4jDriver: neo4j.Driver;
  private pineconeClient: Pinecone;
  //@ts-ignore
  private pineconeIndex: Index;
  private openai: OpenAI;
  private contextCache: SemanticCache<ContextResult>;
  private sessionCache: HybridLRUMRUCache<any[]>;

  constructor() {
    process.stdout.write(`[KNOWLEDGE_BASE] Initializing KnowledgeBase\n`);
    this.openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    process.stdout.write(`[KNOWLEDGE_BASE] OpenAI client initialized\n`);

    this.neo4jDriver = neo4j.driver(
      NEO4J_URI,
      neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD),
    );
    process.stdout.write(
      `[KNOWLEDGE_BASE] Neo4j driver initialized with URI: ${NEO4J_URI}\n`,
    );

    this.pineconeClient = new Pinecone({
      apiKey: PINECONE_API_KEY,
    });
    process.stdout.write(`[KNOWLEDGE_BASE] Pinecone client initialized\n`);

    this.contextCache = new SemanticCache<ContextResult>(
      CONTEXT_CACHE_SIZE,
      this.openai,
    );
    this.sessionCache = new HybridLRUMRUCache<any[]>(SESSION_CACHE_SIZE);

    this.initPineconeClient();
  }

  private async initPineconeClient(): Promise<void> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Initializing Pinecone client and index: ${PINECONE_INDEX_NAME}\n`,
    );
    try {
      const indexesList = await this.pineconeClient.listIndexes();
      process.stdout.write(
        `[KNOWLEDGE_BASE] Retrieved Pinecone indexes: ${JSON.stringify(indexesList.indexes?.map((i) => i.name) || [])}\n`,
      );

      if (!indexesList.indexes?.find((im) => im.name == PINECONE_INDEX_NAME)) {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Index ${PINECONE_INDEX_NAME} not found, creating new index\n`,
        );
        await this.pineconeClient.createIndex({
          name: PINECONE_INDEX_NAME,
          metric: "cosine",
          dimension: parseInt(VECTOR_DIMENSION.toString()),
          spec: {
            serverless: {
              cloud: "aws",
              region: "us-east-1",
            },
          },
        });
        process.stdout.write(
          `[KNOWLEDGE_BASE] Successfully created index: ${PINECONE_INDEX_NAME}\n`,
        );
      } else {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Index ${PINECONE_INDEX_NAME} already exists\n`,
        );
      }

      this.pineconeIndex = this.pineconeClient.Index(PINECONE_INDEX_NAME);
      process.stdout.write(
        `[KNOWLEDGE_BASE] Successfully connected to index: ${PINECONE_INDEX_NAME}\n`,
      );

      await this.initDatabase();
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error initializing Pinecone client: ${error}\n`,
      );
      throw error;
    }
  }

  private async initDatabase(): Promise<void> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Initializing Neo4j database schema\n`,
    );
    const session = this.neo4jDriver.session();
    try {
      process.stdout.write(`[KNOWLEDGE_BASE] Creating user_id constraint\n`);
      await session.run(`
        CREATE CONSTRAINT user_id IF NOT EXISTS 
        FOR (u:User) REQUIRE u.user_id IS UNIQUE
      `);

      process.stdout.write(`[KNOWLEDGE_BASE] Creating session_id constraint\n`);
      await session.run(`
        CREATE CONSTRAINT session_id IF NOT EXISTS 
        FOR (s:Session) REQUIRE s.session_id IS UNIQUE
      `);

      process.stdout.write(`[KNOWLEDGE_BASE] Creating message_id constraint\n`);
      await session.run(`
        CREATE CONSTRAINT message_id IF NOT EXISTS 
        FOR (m:Message) REQUIRE m.message_id IS UNIQUE
      `);

      process.stdout.write(
        `[KNOWLEDGE_BASE] Creating message_vector_id index\n`,
      );
      await session.run(`
        CREATE INDEX message_vector_id IF NOT EXISTS 
        FOR (m:Message) ON (m.vector_id)
      `);

      process.stdout.write(
        `[KNOWLEDGE_BASE] Successfully initialized Neo4j database schema\n`,
      );
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error initializing Neo4j database: ${error}\n`,
      );
      throw error;
    } finally {
      await session.close();
    }
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Generating embedding for text of length: ${text.length}\n`,
    );
    try {
      const response = await this.openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: text,
      });
      process.stdout.write(
        `[KNOWLEDGE_BASE] Successfully generated embedding with dimension: ${response.data[0].embedding.length}\n`,
      );
      return response.data[0].embedding;
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error generating embedding: ${error}\n`,
      );
      throw error;
    }
  }

  public async storeMessage(
    userId: string,
    sessionId: string,
    messageContent: string,
    role: string = "user",
    metadata: RecordMetadata,
  ): Promise<string> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Storing message - userId: ${userId}, sessionId: ${sessionId}, role: ${role}, contentLength: ${messageContent.length}\n`,
    );
    const messageId = uuidv4();
    const vectorId = `vec_${messageId}`;
    const timestamp = Date.now();

    try {
      process.stdout.write(
        `[KNOWLEDGE_BASE] Generating embedding for message: ${messageId}\n`,
      );
      const embedding = await this.generateEmbedding(messageContent);

      process.stdout.write(
        `[KNOWLEDGE_BASE] Upserting vector to Pinecone - vectorId: ${vectorId}, namespace: ${userId}\n`,
      );
      await this.pineconeIndex.namespace(userId).upsert([
        {
          id: vectorId,
          values: embedding,
          metadata: {
            user_id: userId,
            session_id: sessionId,
            message_id: messageId,
            role,
            timestamp,
            ...metadata,
          },
        },
      ]);
      process.stdout.write(
        `[KNOWLEDGE_BASE] Successfully upserted vector to Pinecone\n`,
      );

      const session = this.neo4jDriver.session();
      try {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Storing message in Neo4j - messageId: ${messageId}\n`,
        );
        await session.run(
          `
          MERGE (u:User {user_id: $userId})
          MERGE (s:Session {session_id: $sessionId})
          CREATE (m:Message {
            message_id: $messageId,
            content: $content,
            vector_id: $vectorId,
            role: $role,
            timestamp: $timestamp,
            metadata: $metadataJson
          })
          MERGE (u)-[:PARTICIPATED_IN]->(s)
          MERGE (m)-[:PART_OF]->(s)
          MERGE (u)-[:AUTHORED]->(m)
        `,
          {
            userId,
            sessionId,
            messageId,
            content: messageContent,
            vectorId,
            role,
            timestamp,
            metadataJson: JSON.stringify(metadata),
          },
        );
        process.stdout.write(
          `[KNOWLEDGE_BASE] Successfully stored message in Neo4j\n`,
        );

        process.stdout.write(
          `[KNOWLEDGE_BASE] Invalidating session cache for sessionId: ${sessionId}\n`,
        );
        this.sessionCache.set(sessionId, []);
      } catch (error) {
        process.stderr.write(
          `[KNOWLEDGE_BASE] Error storing message in Neo4j: ${error}\n`,
        );
        throw error;
      } finally {
        await session.close();
      }

      process.stdout.write(
        `[KNOWLEDGE_BASE] Successfully stored message - messageId: ${messageId}\n`,
      );
      return messageId;
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error in storeMessage: ${error}\n`,
      );
      throw error;
    }
  }

  public async getMsgContext(
    userId: string,
    queryText: string,
    topK: number = 5,
  ): Promise<ContextResult> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Getting message context - userId: ${userId}, queryLength: ${queryText.length}, topK: ${topK}\n`,
    );
    try {
      const cacheKey = `context_${userId}_${this.hashString(queryText)}_${topK}`;
      process.stdout.write(
        `[KNOWLEDGE_BASE] Cache key for context: ${cacheKey}\n`,
      );

      const cachedResult = await this.contextCache.get(cacheKey);
      if (cachedResult) {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Returning cached context result with ${cachedResult.messages.length} messages and ${cachedResult.relatedSessions.length} sessions\n`,
        );
        return cachedResult;
      }

      process.stdout.write(`[KNOWLEDGE_BASE] Generating embedding for query\n`);
      const queryEmbedding = await this.generateEmbedding(queryText);

      process.stdout.write(
        `[KNOWLEDGE_BASE] Querying Pinecone for similar vectors - namespace: ${userId}, topK: ${topK}\n`,
      );
      const queryResponse = await this.pineconeIndex.namespace(userId).query({
        vector: queryEmbedding,
        topK,
        includeMetadata: true,
        filter: { user_id: userId },
      });

      const matches = queryResponse.matches || [];
      process.stdout.write(
        `[KNOWLEDGE_BASE] Pinecone returned ${matches.length} matches\n`,
      );

      if (matches.length === 0) {
        process.stdout.write(
          `[KNOWLEDGE_BASE] No matches found, returning empty result\n`,
        );
        const emptyResult = { messages: [], relatedSessions: [] };
        await this.contextCache.set(cacheKey, emptyResult, queryText);
        return emptyResult;
      }

      const vectorIds = matches.map((match) => match.id);
      process.stdout.write(
        `[KNOWLEDGE_BASE] Vector IDs for Neo4j query: ${vectorIds.join(", ")}\n`,
      );

      const session = this.neo4jDriver.session();
      try {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Querying Neo4j for message context\n`,
        );
        const neo4jResult = await session.run(
          `
          MATCH (m:Message)
          WHERE m.vector_id IN $vectorIds
          MATCH (m)-[:PART_OF]->(s:Session)
          WITH s, m
          ORDER BY m.timestamp
          WITH s, collect(m) as sessionMessages
          MATCH (contextMsg:Message)-[:PART_OF]->(s)
          WITH s.session_id as sessionId, contextMsg
          ORDER BY contextMsg.timestamp
          RETURN sessionId, collect({
            content: contextMsg.content,
            role: contextMsg.role,
            timestamp: contextMsg.timestamp,
            metadata: contextMsg.metadata
          }) as contextMessages
        `,
          { vectorIds },
        );
        process.stdout.write(
          `[KNOWLEDGE_BASE] Neo4j returned ${neo4jResult.records.length} session records\n`,
        );

        const sessionMessages: { [key: string]: any[] } = {};
        const sessions: string[] = [];

        neo4jResult.records.forEach((record) => {
          const sessionId = record.get("sessionId");
          const messages = record.get("contextMessages");
          sessionMessages[sessionId] = messages;
          sessions.push(sessionId);
          process.stdout.write(
            `[KNOWLEDGE_BASE] Session ${sessionId} has ${messages.length} messages\n`,
          );
        });

        const allContextMessages = matches.flatMap((match) => {
          const sessionId = match.metadata?.session_id;
          return sessionMessages[sessionId?.toString() || ""]?.map(
            (msg: any) => ({
              ...msg,
              session_id: sessionId,
              metadata: msg.metadata ? JSON.parse(msg.metadata) : undefined,
            }),
          );
        });

        process.stdout.write(
          `[KNOWLEDGE_BASE] Collected ${allContextMessages.length} total context messages across ${sessions.length} unique sessions\n`,
        );

        const contextResult = {
          messages: allContextMessages,
          relatedSessions: [...new Set(sessions)],
        };

        process.stdout.write(`[KNOWLEDGE_BASE] Caching context result\n`);
        await this.contextCache.set(cacheKey, contextResult, queryText);

        return contextResult;
      } catch (error) {
        process.stderr.write(
          `[KNOWLEDGE_BASE] Error querying Neo4j for context: ${error}\n`,
        );
        throw error;
      } finally {
        await session.close();
      }
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error in getMsgContext: ${error}\n`,
      );
      throw error;
    }
  }

  public async getSessionHistory(sessionId: string): Promise<any[]> {
    process.stdout.write(
      `[KNOWLEDGE_BASE] Getting session history - sessionId: ${sessionId}\n`,
    );
    try {
      const cachedResult = this.sessionCache.get(sessionId);
      if (cachedResult) {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Returning cached session history with ${cachedResult.length} messages\n`,
        );
        return cachedResult;
      }

      const session = this.neo4jDriver.session();
      try {
        process.stdout.write(
          `[KNOWLEDGE_BASE] Querying Neo4j for session history\n`,
        );
        const result = await session.run(
          `
          MATCH (m:Message)-[:PART_OF]->(:Session {session_id: $sessionId})
          RETURN m.content as content, m.role as role, m.timestamp as timestamp,
                 m.metadata as metadata
          ORDER BY m.timestamp
        `,
          { sessionId },
        );
        process.stdout.write(
          `[KNOWLEDGE_BASE] Neo4j returned ${result.records.length} messages for session\n`,
        );

        const messages = result.records.map((record) => {
          const content = record.get("content");
          const metadata = record.get("metadata");
          process.stdout.write(
            `[KNOWLEDGE_BASE] Processing message with content length: ${content.length}, has metadata: ${!!metadata}\n`,
          );
          return {
            content: content,
            role: record.get("role"),
            timestamp: record.get("timestamp"),
            metadata: metadata ? JSON.parse(metadata) : {},
          };
        });

        process.stdout.write(
          `[KNOWLEDGE_BASE] Caching session history with ${messages.length} messages\n`,
        );
        this.sessionCache.set(sessionId, messages);

        return messages;
      } catch (error) {
        process.stderr.write(
          `[KNOWLEDGE_BASE] Error querying Neo4j for session history: ${error}\n`,
        );
        throw error;
      } finally {
        await session.close();
      }
    } catch (error) {
      process.stderr.write(
        `[KNOWLEDGE_BASE] Error in getSessionHistory: ${error}\n`,
      );
      throw error;
    }
  }

  public getCacheStats(): {
    contextCache: {
      hits: number;
      misses: number;
      size: number;
      hitRate: number;
    };
    sessionCache: {
      hits: number;
      misses: number;
      size: number;
      hitRate: number;
    };
  } {
    process.stdout.write(`[KNOWLEDGE_BASE] Getting cache stats\n`);
    const contextStats = this.contextCache.getStats();
    const sessionStats = this.sessionCache.getStats();
    process.stdout.write(
      `[KNOWLEDGE_BASE] Context cache stats: ${JSON.stringify(contextStats)}\n`,
    );
    process.stdout.write(
      `[KNOWLEDGE_BASE] Session cache stats: ${JSON.stringify(sessionStats)}\n`,
    );
    return {
      contextCache: contextStats,
      sessionCache: sessionStats,
    };
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash |= 0;
    }
    return hash.toString(16);
  }

  public async close(): Promise<void> {
    process.stdout.write(`[KNOWLEDGE_BASE] Closing Neo4j driver connection\n`);
    await this.neo4jDriver.close();
    process.stdout.write(
      `[KNOWLEDGE_BASE] Successfully closed Neo4j driver connection`,
    );
  }
}
