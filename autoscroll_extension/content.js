// Checks if the section with class 'blocks-lesson' is visible in the DOM.
function isSectionVisible() {
    const section = document.querySelector("section.blocks-lesson");
    return section !== null;
}

// Scrolls to the target div with the specified block ID and highlights it.
function scrollToDiv(blockId) {
    const div = document.querySelector(`[data-block-id="${blockId}"]`);
    if (div) {
        // Smoothly scroll to the div
        div.scrollIntoView({ behavior: "smooth", block: "center" });

        // Highlight the div with a red line
        div.style.border = "2px solid red";
        div.style.padding = "5px";
    } else {
        console.error(`Div with data-block-id="${blockId}" not found!`);
    }
}

// Extracts the block ID from the URL hash.
function extractBlockIdFromHash() {
    const hash = window.location.hash;
    // Match /data-block-id/ followed by any characters
    const blockIdMatch = hash.match(/\/block\/(.+)/);
    return blockIdMatch ? blockIdMatch[1] : null;
}

// Waits for the section to become visible and then scrolls to the target div.
function waitAndScrollToDiv(blockId, timeout = 10000) {
    const startTime = Date.now();

    // Start checking for the section immediately
    const checkInterval = setInterval(() => {
        if (isSectionVisible()) {
            clearInterval(checkInterval); // Stop checking once the section is visible
            scrollToDiv(blockId);
        } else if (Date.now() - startTime >= timeout) {
            clearInterval(checkInterval); // Stop checking after timeout
            console.error("Section not found within the timeout period!");
        }
    }, 500); // Check every 500ms
}

const blockId = extractBlockIdFromHash();

if (blockId) {
    waitAndScrollToDiv(blockId);
}
