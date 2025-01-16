// Inject CSS for iframe
const style = document.createElement("style");
style.textContent = `
    .embed-iframe {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 100px;
        height: 100px;
        z-index: 9999;
        border: none;
    }
    .embed-iframe.active {
        width: 500px;
        height: 1000px;
    }
`;
document.head.appendChild(style);

// Create and inject iframe
const iframe = document.createElement("iframe");
iframe.src = "http://127.0.0.1:5500/RAG/chainlit/chainlit_copilot.html";
iframe.className = "embed-iframe";
document.body.appendChild(iframe);

// Add event listeners for interaction
iframe.addEventListener("mouseenter", () => {
    iframe.classList.add("active");
});

document.addEventListener("click", (event) => {
    if (!iframe.contains(event.target)) {
        iframe.classList.remove("active");
    }
});
