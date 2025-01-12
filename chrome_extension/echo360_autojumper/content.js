// Extracts timestamp from the URL query (e.g., ?t=120)
function extractTimestampFromQuery() {
    const urlParams = new URLSearchParams(window.location.search);
    const timestamp = urlParams.get("t"); // e.g., "120" for 120 seconds
    return timestamp ? parseFloat(timestamp) : null;
}

// Extracts the duration from the JSON-LD script tag
function extractDuration() {
    const jsonLdScript = document.querySelector(
        'script[type="application/ld+json"]'
    );

    if (jsonLdScript) {
        try {
            const videoMetadata = JSON.parse(jsonLdScript.textContent);
            return videoMetadata.duration;
        } catch (error) {
            console.error("Error parsing JSON-LD:", error);
        }
    } else {
        console.error("JSON-LD script tag not found.");
    }

    return null;
}

// Converts ISO 8601 duration (PT<seconds>S) to seconds
function isoDurationToSeconds(duration) {
    const seconds = parseFloat(duration.replace("PT", "").replace("S", ""));

    if (isNaN(seconds)) {
        console.error("Invalid duration format:", duration);
        return null;
    }

    return seconds;
}

// Checks if the timeline progress bar is visible in the DOM
function isProgressBarVisible() {
    const progressBar = document.querySelector("div#timeline-progress-bar");
    return progressBar !== null;
}

// Jumps to the specified timestamp in the video
function jumpToTimestamp(timestamp) {
    if (isProgressBarVisible()) {
        const progressBar = document.querySelector("div#timeline-progress-bar");
        const progressBarWidth = progressBar.offsetWidth;

        // Calculate the click position based on the timestamp and total duration
        const duration = isoDurationToSeconds(extractDuration());
        if (!duration) {
            console.error("Failed to extract video duration.");
            return;
        }

        const percentage = (timestamp / duration) * 100;
        const clickPosition = (progressBarWidth * percentage) / 100;

        // Simulate a click on the progress bar
        const clickEvent = new MouseEvent("click", {
            bubbles: true,
            clientX: progressBar.offsetLeft + clickPosition,
            clientY: progressBar.offsetTop + progressBar.offsetHeight / 2,
        });

        progressBar.dispatchEvent(clickEvent);

        console.log(
            `Jumped to ${timestamp} seconds (${percentage.toFixed(2)}%)`
        );
    } else {
        console.error("Timeline progress bar is not visible.");
    }
}

// Waits for the timeline progress bar to appear and then jumps to the timestamp
function waitAndJumpToTimestamp(timestamp, timeout = 10000) {
    const startTime = Date.now();

    // Start checking for the progress bar
    const checkInterval = setInterval(() => {
        if (isProgressBarVisible()) {
            clearInterval(checkInterval); // Stop checking once the progress bar is visible
            jumpToTimestamp(timestamp); // Jump to the timestamp
        } else if (Date.now() - startTime >= timeout) {
            clearInterval(checkInterval); // Stop checking after timeout
            console.error("Progress bar not found within the timeout period!");
        }
    }, 500); // Check every 500ms
}

// Extract timestamp from the URL
const timestamp = extractTimestampFromQuery();

if (timestamp !== null) {
    // Wait for the progress bar to appear and then jump to the timestamp
    waitAndJumpToTimestamp(timestamp);
}
