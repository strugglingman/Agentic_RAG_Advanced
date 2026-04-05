-- Add channel-scoped conversation key for Slack/Teams lookup.
ALTER TABLE "Conversation"
ADD COLUMN IF NOT EXISTS "source_channel_id" TEXT;

-- Composite index used by channel integrations to find/reuse threads quickly.
CREATE INDEX IF NOT EXISTS "Conversation_user_email_source_channel_id_idx"
ON "Conversation"("user_email", "source_channel_id");
