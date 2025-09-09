import { genkit, z } from "genkit";
import { chroma } from "genkitx-chromadb";
import { chromaRetrieverRef } from 'genkitx-chromadb';
import { chromaIndexerRef } from 'genkitx-chromadb';

// Simple text embedder function
const simpleTextEmbedder = {
  name: "simple-text-embedder",
  embed: async (texts: string[]) => {
    // Simple hash-based embedding for demonstration
    // In a real application, you'd want to use a proper embedding model
    return texts.map(text => {
      // Create a simple vector based on text characteristics
      const vector = new Array(384).fill(0);
      const words = text.toLowerCase().split(/\s+/);
      
      // Simple word-based features
      words.forEach((word, index) => {
        const hash = word.split('').reduce((a, b) => {
          a = ((a << 5) - a) + b.charCodeAt(0);
          return a & a;
        }, 0);
        const pos = Math.abs(hash) % 384;
        vector[pos] += 1 / (index + 1);
      });
      
      // Normalize the vector
      const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
      return vector.map(val => val / magnitude);
    });
  }
};

// Initialize Genkit with ChromaDB plugin only
const ai = genkit({
  plugins: [
    chroma([
      {
        collectionName: "policies",
        embedder: simpleTextEmbedder,
        clientParams: {
          path: "http://localhost:8000",
        },
      },
    ]),
  ],
});

// Define input schema
const RecipeInputSchema = z.object({
  ingredient: z.string().describe("Main ingredient or cuisine type"),
  dietaryRestrictions: z
    .string()
    .optional()
    .describe("Any dietary restrictions"),
});

// Define output schema
const RecipeSchema = z.object({
  title: z.string(),
  description: z.string(),
  prepTime: z.string(),
  cookTime: z.string(),
  servings: z.number(),
  ingredients: z.array(z.string()),
  instructions: z.array(z.string()),
  tips: z.array(z.string()).optional(),
});

// Create specific retriever and indexer for policies collection
export const policiesRetriever = chromaRetrieverRef({
  collectionName: 'policies',
});

export const policiesIndexer = chromaIndexerRef({
  collectionName: 'policies',
});
 
// Example function to retrieve similar policies
async function retrieveSimilarPolicies(query: string) {
  // Use the configured retriever
  const docs = await ai.retrieve({ 
    retriever: chromaRetrieverRef, 
    query 
  });

  console.log(`Found ${docs.length} similar policies for query: "${query}"`);
  docs.forEach((doc, index) => {
    const content = doc.content?.[0]?.text || doc.text || "No content available";
    console.log(`${index + 1}. ${content.substring(0, 150)}...`);
    if (doc.metadata) {
      console.log(`   Metadata: ${JSON.stringify(doc.metadata)}`);
    }
  });

  return docs;
}

// Define input schema for policy queries
const PolicyQueryInputSchema = z.object({
  query: z.string().describe("Policy question or topic to search for"),
});

// Define output schema for policy responses
const PolicyResponseSchema = z.object({
  answer: z.string().describe("Comprehensive answer based on relevant policies"),
  relevantPolicies: z.array(z.string()).describe("List of relevant policy excerpts"),
  policyTypes: z.array(z.string()).describe("Types of policies referenced"),
});

export const policyQueryFlow = ai.defineFlow(
  {
    name: "policyQueryFlow",
    inputSchema: PolicyQueryInputSchema,
    outputSchema: PolicyResponseSchema,
  },
  async (input) => {
    // First, retrieve relevant policies from the database
    const relevantPolicies = await retrieveSimilarPolicies(input.query);

    // Extract policy content and metadata
    const policyTexts = relevantPolicies.map((doc) => {
      return doc.content?.[0]?.text || doc.text || "No content available";
    });

    const policyTypes = relevantPolicies
      .map((doc) => doc.metadata?.policyType)
      .filter(Boolean)
      .filter((value, index, self) => self.indexOf(value) === index); // Remove duplicates

    // Create a simple response without AI generation
    const answer = `Based on your query "${input.query}", here are the relevant policies:

${policyTexts.map((text, i) => `${i + 1}. ${text}`).join('\n\n')}

Please review these policies for the specific information you need.`;

    return {
      answer,
      relevantPolicies: policyTexts,
      policyTypes: policyTypes as string[],
    };
  }
);
 