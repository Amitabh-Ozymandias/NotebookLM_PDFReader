import "dotenv/config";

import express from "express";
import cors from "cors";
import multer from "multer";
import fs from "fs";
import pdf from "pdf-parse";

import {
    GoogleGenerativeAI
} from "@google/generative-ai";


// -------------------------------------
// APP SETUP
// -------------------------------------

const app = express();

app.use(cors({
    origin: "*"
}));

app.use(express.json());


// -------------------------------------
// UPLOADS FOLDER
// -------------------------------------

if (!fs.existsSync("uploads")) {
    fs.mkdirSync("uploads");
}

const upload = multer({
    dest: "uploads/",
});


// -------------------------------------
// GEMINI
// -------------------------------------

const genAI = new GoogleGenerativeAI(
    process.env.GEMINI_API_KEY
);


// -------------------------------------
// MEMORY STORE
// -------------------------------------

let documentChunks = [];


// -------------------------------------
// CHUNKING
// -------------------------------------

function chunkText(
    text,
    chunkSize = 1200,
    overlap = 200
) {

    const chunks = [];

    for (
        let i = 0;
        i < text.length;
        i += chunkSize - overlap
    ) {

        chunks.push(
            text.slice(i, i + chunkSize)
        );
    }

    return chunks;
}


// -------------------------------------
// LOCAL EMBEDDINGS
// -------------------------------------

function createEmbedding(text) {

    const words =
        text
            .toLowerCase()
            .split(/\W+/)
            .filter(Boolean);

    const vector = {};

    for (const word of words) {

        vector[word] =
            (vector[word] || 0) + 1;
    }

    return vector;
}


// -------------------------------------
// COSINE SIMILARITY
// -------------------------------------

function cosineSimilarity(vecA, vecB) {

    const allWords =
        new Set([
            ...Object.keys(vecA),
            ...Object.keys(vecB),
        ]);

    let dot = 0;

    let magA = 0;

    let magB = 0;

    for (const word of allWords) {

        const a = vecA[word] || 0;

        const b = vecB[word] || 0;

        dot += a * b;

        magA += a * a;

        magB += b * b;
    }

    magA = Math.sqrt(magA);

    magB = Math.sqrt(magB);

    if (magA === 0 || magB === 0) {
        return 0;
    }

    return dot / (magA * magB);
}


// -------------------------------------
// HEALTH CHECK
// -------------------------------------

app.get("/", (req, res) => {

    res.json({
        success: true,
        message: "RAG Backend Running",
    });
});


// -------------------------------------
// PDF UPLOAD
// -------------------------------------

app.post(
    "/upload",
    upload.single("pdf"),
    async (req, res) => {

        try {

            if (!req.file) {

                return res.status(400).json({
                    success: false,
                    error: "No PDF uploaded",
                });
            }

            // clear previous doc
            documentChunks = [];

            // read uploaded pdf
            const dataBuffer =
                fs.readFileSync(req.file.path);

            const pdfData =
                await pdf(dataBuffer);

            if (!pdfData.text) {

                return res.status(400).json({
                    success: false,
                    error: "Could not extract text from PDF",
                });
            }

            // chunk document
            const chunks =
                chunkText(pdfData.text);

            console.log(
                `Chunks created: ${chunks.length}`
            );

            // generate local embeddings
            for (const chunk of chunks) {

                const embedding =
                    createEmbedding(chunk);

                documentChunks.push({
                    text: chunk,
                    embedding,
                });
            }

            // delete temp upload
            fs.unlinkSync(req.file.path);

            console.log(
                "PDF indexed successfully"
            );

            res.json({
                success: true,
                message:
                    `Successfully indexed ${chunks.length} chunks.`,
            });

        } catch (error) {

            console.error(
                "Upload Error:",
                error
            );

            res.status(500).json({
                success: false,
                error: error.message,
            });
        }
    }
);


// -------------------------------------
// QUESTION ANSWERING
// -------------------------------------

app.post("/ask", async (req, res) => {

    try {

        const { question } = req.body;

        if (!question) {

            return res.status(400).json({
                success: false,
                error: "Question is required",
            });
        }

        if (documentChunks.length === 0) {

            return res.status(400).json({
                success: false,
                error: "Upload a PDF first",
            });
        }

        // create question embedding
        const questionEmbedding =
            createEmbedding(question);

        // similarity search
        const scoredChunks =
            documentChunks.map(chunk => ({

                text: chunk.text,

                score: cosineSimilarity(
                    questionEmbedding,
                    chunk.embedding
                ),
            }));


        // sort best first
        scoredChunks.sort(
            (a, b) => b.score - a.score
        );


        // retrieve top chunks
        const topChunks =
            scoredChunks
                .slice(0, 3)
                .map(c => c.text)
                .join("\n\n");


        // Gemini answer generation
        const model =
            genAI.getGenerativeModel({
                model: "gemini-2.5-flash-lite",
            });


        const prompt = `
You are a RAG assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:
"I could not find that in the uploaded document."

Context:
${topChunks}

Question:
${question}
`;


        const result =
            await model.generateContent(prompt);

        const answer =
            result.response.text();


        res.json({
            success: true,
            answer,
        });

    } catch (error) {

        console.error(
            "Question Error:",
            error
        );

        res.status(500).json({
            success: false,
            error: error.message,
        });
    }
});


// -------------------------------------
// START SERVER
// -------------------------------------

const PORT =
    process.env.PORT || 5000;

app.listen(PORT, () => {

    console.log(
        `Server running on port ${PORT}`
    );
});