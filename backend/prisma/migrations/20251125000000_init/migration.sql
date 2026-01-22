-- ============================================================================
-- Initial Migration: All Tables
-- ============================================================================

-- ============================================================================
-- User Table (must be first - other tables reference it)
-- ============================================================================
CREATE TABLE "User" (
    "id" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "passwordHash" TEXT NOT NULL,
    "name" TEXT,
    "dept" TEXT,
    "role" TEXT NOT NULL DEFAULT 'staff',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "User_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "User_email_key" ON "User"("email");

-- ============================================================================
-- Conversation Table
-- ============================================================================
CREATE TABLE "Conversation" (
    "id" TEXT NOT NULL,
    "user_email" TEXT NOT NULL,
    "title" TEXT,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "Conversation_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "Conversation_user_email_idx" ON "Conversation"("user_email");
CREATE INDEX "Conversation_updated_at_idx" ON "Conversation"("updated_at");

-- ============================================================================
-- Message Table
-- ============================================================================
CREATE TABLE "Message" (
    "id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "role" TEXT NOT NULL,
    "content" TEXT NOT NULL,
    "tokens_used" INTEGER,
    "latency_ms" INTEGER,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Message_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "Message_conversation_id_idx" ON "Message"("conversation_id");
CREATE INDEX "Message_created_at_idx" ON "Message"("created_at");

-- ============================================================================
-- QueryLog Table
-- ============================================================================
CREATE TABLE "QueryLog" (
    "id" TEXT NOT NULL,
    "user_email" TEXT NOT NULL,
    "conversation_id" TEXT,
    "query" TEXT NOT NULL,
    "response_time_ms" INTEGER NOT NULL,
    "tokens_used" INTEGER NOT NULL,
    "retrieval_quality" DOUBLE PRECISION,
    "recommendation" TEXT,
    "refinement_count" INTEGER NOT NULL DEFAULT 0,
    "contexts_retrieved" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "QueryLog_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "QueryLog_user_email_idx" ON "QueryLog"("user_email");
CREATE INDEX "QueryLog_created_at_idx" ON "QueryLog"("created_at");
CREATE INDEX "QueryLog_recommendation_idx" ON "QueryLog"("recommendation");

-- ============================================================================
-- FileRegistry Table
-- ============================================================================
CREATE TABLE "FileRegistry" (
    "id" TEXT NOT NULL,
    "user_email" TEXT NOT NULL,
    "dept_id" TEXT,
    "category" TEXT NOT NULL,
    "original_name" TEXT NOT NULL,
    "storage_path" TEXT NOT NULL,
    "download_url" TEXT,
    "mime_type" TEXT,
    "size_bytes" BIGINT,
    "source_tool" TEXT,
    "source_url" TEXT,
    "conversation_id" TEXT,
    "indexed_in_chromadb" BOOLEAN NOT NULL DEFAULT false,
    "chromadb_collection" TEXT,
    "metadata" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "accessed_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "FileRegistry_pkey" PRIMARY KEY ("id")
);

CREATE INDEX "FileRegistry_user_email_category_idx" ON "FileRegistry"("user_email", "category");
CREATE INDEX "FileRegistry_user_email_created_at_idx" ON "FileRegistry"("user_email", "created_at");
CREATE INDEX "FileRegistry_conversation_id_idx" ON "FileRegistry"("conversation_id");
CREATE INDEX "FileRegistry_category_idx" ON "FileRegistry"("category");

-- ============================================================================
-- Foreign Keys
-- ============================================================================
ALTER TABLE "Conversation" ADD CONSTRAINT "Conversation_user_email_fkey"
    FOREIGN KEY ("user_email") REFERENCES "User"("email") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "Message" ADD CONSTRAINT "Message_conversation_id_fkey"
    FOREIGN KEY ("conversation_id") REFERENCES "Conversation"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "QueryLog" ADD CONSTRAINT "QueryLog_user_email_fkey"
    FOREIGN KEY ("user_email") REFERENCES "User"("email") ON DELETE CASCADE ON UPDATE CASCADE;
