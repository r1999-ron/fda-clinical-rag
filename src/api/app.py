from fastapi import (FastAPI, HTTPException,)

from src.api.models import (AskRequest,AskResponse,DocumentResponse,HealthResponse,)

from src.api.session_store import (session_store,)

from src.conversation.conversational_rag import (generate_conversational_answer,)

from src.conversation.models import (
    ConversationMessage,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import HTMLResponse

from src.indexing.manifest import (load_manifest,)


app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description=(
        "Enterprise RAG service with "
        "incremental indexing, metadata filtering, "
        "conversational query rewriting, "
        "grounded generation, and citations."
    ),
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/", response_class=HTMLResponse,)
def home():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>Enterprise RAG Assistant</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;

            background: #f5f7fa;
            color: #1f2937;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 24px;
        }

        .header {
            margin-bottom: 20px;
        }

        .header h1 {
            margin-bottom: 6px;
        }

        .header p {
            margin-top: 0;
            color: #6b7280;
        }

        .controls {
            background: white;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 16px;

            box-shadow:
                0 1px 3px
                rgba(0, 0, 0, 0.08);
        }

        .controls label {
            display: block;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .controls select {
            width: 100%;
            padding: 10px;
            border:
                1px solid #d1d5db;

            border-radius: 8px;
        }

        .chat {
            height: 550px;
            overflow-y: auto;

            background: white;
            padding: 20px;

            border-radius: 12px;

            box-shadow:
                0 1px 3px
                rgba(0, 0, 0, 0.08);

            margin-bottom: 16px;
        }

        .message {
            margin-bottom: 18px;
        }

        .user {
            text-align: right;
        }

        .bubble {
    display: inline-block;
    max-width: 80%;
    padding: 14px 16px;

    border-radius: 14px;

    white-space: normal;
    line-height: 1.6;

    text-align: left;
}

.answer-text {
    white-space: pre-line;
}

.metadata {
    margin-top: 12px;
    padding-top: 10px;

    border-top: 1px solid #e5e7eb;

    font-size: 12px;
    color: #6b7280;
}

.citations {
    margin-top: 12px;
    padding-top: 10px;

    border-top: 1px solid #e5e7eb;
}

.citations-title {
    font-weight: 600;
    margin-bottom: 6px;
}

.citation {
    margin-top: 4px;
    font-size: 13px;
    color: #4b5563;
}

        .user .bubble {
            background: #2563eb;
            color: white;
        }

        .assistant .bubble {
            background: #f3f4f6;
            color: #111827;
        }

        .metadata {
            margin-top: 8px;

            font-size: 12px;
            color: #6b7280;
        }

        .citations {
            margin-top: 10px;

            padding-top: 8px;

            border-top:
                1px solid #e5e7eb;

            font-size: 13px;
        }

        .citation {
            margin-top: 5px;
        }

        .input-row {
            display: flex;
            gap: 10px;
        }

        #question {
            flex: 1;

            padding: 14px;

            border:
                1px solid #d1d5db;

            border-radius: 10px;

            font-size: 15px;
        }

        button {
            padding: 0 22px;

            background: #111827;
            color: white;

            border: none;
            border-radius: 10px;

            cursor: pointer;
            font-size: 15px;
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .loading {
            color: #6b7280;
            font-style: italic;
        }

    </style>
</head>

<body>

<div class="container">

    <div class="header">

        <h1>Enterprise RAG Assistant</h1>

        <p>
            Ask questions across your indexed regulatory documents.
        </p>

    </div>


    <div class="controls">

        <label for="documentFilter">
            Search scope
        </label>

        <select id="documentFilter">

            <option value="">
                All documents
            </option>

        </select>

    </div>


    <div
        id="chat"
        class="chat"
    ></div>


    <div class="input-row">

        <input
            id="question"
            type="text"
            placeholder="Ask a question..."
        >

        <button
            id="sendButton"
            onclick="sendQuestion()"
        >
            Send
        </button>

    </div>

</div>


<script>

    const sessionId =
        "web-" +
        crypto.randomUUID();

    const chat =
        document.getElementById(
            "chat"
        );

    const questionInput =
        document.getElementById(
            "question"
        );

    const sendButton =
        document.getElementById(
            "sendButton"
        );

    const documentFilter =
        document.getElementById(
            "documentFilter"
        );


    async function loadDocuments() {

    const response =
        await fetch(
            "/documents"
        );

    const documents =
        await response.json();


    documents.forEach(
        doc => {

            const option =
                window.document.createElement(
                    "option"
                );

            option.value =
                doc.document_id;

            option.textContent =
                doc.filename;

            documentFilter.appendChild(
                option
            );

        }
    );

}


    function addUserMessage(
    text
) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message user";


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";

    bubble.textContent =
        text;


    wrapper.appendChild(
        bubble
    );

    chat.appendChild(
        wrapper
    );

    scrollChat();

}


    function addAssistantMessage(data) {

    const wrapper =
        document.createElement(
            "div"
        );

    wrapper.className =
        "message assistant";


    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";


    // ==========================================
    // ANSWER
    // ==========================================

    const answer =
        document.createElement(
            "div"
        );

    answer.className =
        "answer-text";

    answer.textContent =
        data.answer;

    bubble.appendChild(
        answer
    );


    // ==========================================
    // DEBUG / RETRIEVAL METADATA
    // ==========================================

    const metadata =
        document.createElement(
            "div"
        );

    metadata.className =
        "metadata";


    const retrievalQuery =
        document.createElement(
            "div"
        );

    retrievalQuery.textContent =
        "Retrieval query: "
        + data.retrieval_query;


    const rewritten =
        document.createElement(
            "div"
        );

    rewritten.textContent =
        "Rewritten: "
        + data.query_rewritten;


    metadata.appendChild(
        retrievalQuery
    );

    metadata.appendChild(
        rewritten
    );

    bubble.appendChild(
        metadata
    );


    // ==========================================
    // CITATIONS
    // ==========================================

    const citations =
        data.citations || {};

    const citationEntries =
        Object.entries(
            citations
        );


    if (
        citationEntries.length > 0
    ) {

        const citationContainer =
            document.createElement(
                "div"
            );

        citationContainer.className =
            "citations";


        const title =
            document.createElement(
                "div"
            );

        title.className =
            "citations-title";

        title.textContent =
            "Sources";

        citationContainer.appendChild(
            title
        );


        citationEntries.forEach(
            ([id, citation]) => {

                const citationElement =
                    document.createElement(
                        "div"
                    );

                citationElement.className =
                    "citation";


                let text =
                    `[${id}] `
                    + citation.source;


                if (
                    citation.page
                    !== null
                    && citation.page
                    !== undefined
                ) {

                    text +=
                        ` — Page ${citation.page}`;

                }


                citationElement.textContent =
                    text;


                citationContainer.appendChild(
                    citationElement
                );

            }
        );


        bubble.appendChild(
            citationContainer
        );

    }


    wrapper.appendChild(
        bubble
    );

    chat.appendChild(
        wrapper
    );

    scrollChat();

}


    function addLoadingMessage() {

        const wrapper =
            document.createElement(
                "div"
            );

        wrapper.id =
            "loading-message";

        wrapper.className =
            "message assistant";

        wrapper.innerHTML =
            `
            <div
                class="bubble loading"
            >
                Searching documents...
            </div>
            `;

        chat.appendChild(
            wrapper
        );

        scrollChat();

    }


    function removeLoadingMessage() {

        const loading =
            document.getElementById(
                "loading-message"
            );

        if (loading) {
            loading.remove();
        }

    }


    async function sendQuestion() {

        const question =
            questionInput
            .value
            .trim();


        if (!question) {
            return;
        }


        addUserMessage(
            question
        );


        questionInput.value =
            "";


        sendButton.disabled =
            true;


        addLoadingMessage();


        const selectedDocument =
            documentFilter.value;


        const payload = {

            session_id:
                sessionId,

            question:
                question,

            filters:
                selectedDocument
                ? {
                    document_id:
                        selectedDocument
                }
                : null

        };


        try {

            const response =
                await fetch(
                    "/ask",
                    {
                        method:
                            "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                payload
                            )
                    }
                );


            removeLoadingMessage();


            if (!response.ok) {

                const error =
                    await response.json();

                throw new Error(
                    error.detail
                    || "Request failed"
                );

            }


            const data =
                await response.json();


            addAssistantMessage(
                data
            );


        } catch (error) {

            removeLoadingMessage();


            const wrapper =
                document.createElement(
                    "div"
                );

            wrapper.className =
                "message assistant";


            wrapper.innerHTML =
                `
                <div class="bubble">
                    Error:
                    ${escapeHtml(
                        error.message
                    )}
                </div>
                `;


            chat.appendChild(
                wrapper
            );

        } finally {

            sendButton.disabled =
                false;

            questionInput.focus();

        }

    }


    function escapeHtml(
        text
    ) {

        const div =
            document.createElement(
                "div"
            );

        div.textContent =
            text;

        return div.innerHTML;

    }


    function scrollChat() {

        chat.scrollTop =
            chat.scrollHeight;

    }


    questionInput.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Enter"
            ) {

                sendQuestion();

            }

        }
    );


    loadDocuments();

</script>

</body>

</html>
"""
@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="ok"
    )


# ============================================================
# DOCUMENTS
# ============================================================

@app.get("/documents",response_model=list[DocumentResponse],)
def list_documents():
    manifest = load_manifest()
    documents = (
        manifest.get(
            "documents",
            {}
        )
    )
    response = []

    for (
        document_id,
        metadata,
    ) in documents.items():

        response.append(
            DocumentResponse(
                document_id=document_id,
                filename=metadata.get(
                    "filename",
                    document_id,
                ),
                chunk_count=metadata.get(
                    "chunk_count",
                    0,
                ),
                indexed_at=metadata.get(
                    "indexed_at"
                ),
            )
        )

    return response


# ============================================================
# ASK
# ============================================================

@app.post("/ask", response_model=AskResponse,)
def ask(request: AskRequest,):
    try:
        history = (
            session_store
            .get_history(
                request.session_id
            )
        )
        result = (
            generate_conversational_answer(
                question=request.question,
                history=history,
                filters=request.filters,
                retrieval_k=5,
                context_k=5,
                session_id=request.session_id,
            )
        )
        # Store original user message,
        # NOT rewritten retrieval query.
        session_store.add_turn(
            session_id=request.session_id,
            user_message=request.question,
            assistant_message=result[
                "answer"
            ],
        )
        return AskResponse(
            session_id=request.session_id,

            original_question=result[
                "original_question"
            ],

            retrieval_query=result[
                "retrieval_query"
            ],

            query_rewritten=result[
                "query_rewritten"
            ],

            answer=result[
                "answer"
            ],

            citations=result.get(
                "citations",
                {},
            ),

            citation_validation=result.get(
                "citation_validation",
                {},
            ),

            trace=result.get(
                "trace",
                {},
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# ============================================================
# SESSION HISTORY
# ============================================================

@app.get("/sessions/{session_id}",)
def get_session(session_id: str,):

    history = (
        session_store
        .get_history(
            session_id
        )
    )

    return {
        "session_id": (
            session_id
        ),
        "messages": [
            message.model_dump()
            for message in history
        ],
    }


# ============================================================
# CLEAR SESSION
# ============================================================

@app.delete("/sessions/{session_id}",)
def clear_session(session_id: str,):
    session_store.clear(
        session_id
    )
    return {
        "session_id": (
            session_id
        ),
        "cleared": True,
    }