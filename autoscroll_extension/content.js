// Extracts multiple block IDs from the URL hash (separated by commas).
function extractBlockIdsFromHash() {
    const hash = window.location.hash;
    // Match /block/ followed by any characters (including commas)
    const blockIdsMatch = hash.match(/\/block\/([^#]+)/);
    return blockIdsMatch ? blockIdsMatch[1].split(",") : [];
}

// Checks if the section with class 'blocks-lesson' is visible in the DOM.
function isSectionVisible() {
    const section = document.querySelector("section.blocks-lesson");
    return section !== null;
}

// Highlights all blocks from the blockIds array
function highlightBlocks(blockIds) {
    if (blockIds.length > 0) {
        // Highlight all specified blocks with red line
        blockIds.forEach((blockId) => {
            const div = document.querySelector(`[data-block-id="${blockId}"]`);
            if (div) {
                div.style.border = "2px solid red";
                div.style.padding = "5px";
            } else {
                console.error(`Div with data-block-id="${blockId}" not found!`);
            }
        });
    } else {
        console.error("No valid block IDs found.");
    }
}

// Scrolls to the target div with the specified block ID.
function scrollToDiv(blockId) {
    const div = document.querySelector(`[data-block-id="${blockId}"]`);
    if (div) {
        // Smoothly scroll to the div
        div.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
        console.error(`Div with data-block-id="${blockId}" not found!`);
    }
}

// Set up the "Next" button to navigate through the blocks.
function setupNavigation(blockIds) {
    let currentBlockIndex = 0;

    // Create the "Next" button
    const nextButton = document.createElement("button");
    nextButton.textContent = "Next Highlight";
    nextButton.style.position = "fixed";
    nextButton.style.top = "20px";
    nextButton.style.right = "20px";
    nextButton.style.padding = "10px";
    nextButton.style.backgroundColor = "red";
    nextButton.style.color = "#fff";
    nextButton.style.border = "none";
    nextButton.style.cursor = "pointer";
    nextButton.style.zIndex = "1000";
    document.body.appendChild(nextButton);

    // Event listener to scroll to the next block when clicked
    nextButton.addEventListener("click", () => {
        if (blockIds.length > 0) {
            // Scroll to the next block
            scrollToDiv(blockIds[currentBlockIndex]);

            // Update the currentBlockIndex, looping back to the start if necessary
            currentBlockIndex = (currentBlockIndex + 1) % blockIds.length;
        }
    });
}

// Waits for the section to become visible and then scrolls to the target div.
function waitAndHighlightDiv(blockIds, timeout = 10000) {
    const startTime = Date.now();
    // Start checking for the section immediately
    const checkInterval = setInterval(() => {
        if (isSectionVisible()) {
            clearInterval(checkInterval); // Stop checking once the section is visible
            highlightBlocks(blockIds); // Highlight all blocks
            setupNavigation(blockIds); // Setup navigation after highlighting blocks
            scrollToDiv(blockIds[0]); // Scroll to the first block
        } else if (Date.now() - startTime >= timeout) {
            clearInterval(checkInterval); // Stop checking after timeout
            console.error("Section not found within the timeout period!");
        }
    }, 500); // Check every 500ms
}

// Extract block IDs from the URL hash
const blockIds = extractBlockIdsFromHash();

if (blockIds.length > 0) {
    waitAndHighlightDiv(blockIds);
}
