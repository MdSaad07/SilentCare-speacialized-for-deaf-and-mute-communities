    const video = document.getElementById('video');
    const canvas = document.getElementById('skeleton');
    const ctx = canvas.getContext('2d');
    const currentChar = document.getElementById('current-char');
    const sentenceSpan = document.getElementById('sentence');
    const suggButtons = [
        document.getElementById('sugg1'),
        document.getElementById('sugg2'),
        document.getElementById('sugg3'),
        document.getElementById('sugg4')
    ];
    const speakBtn = document.getElementById('speak-btn');
    const clearBtn = document.getElementById('clear-btn');
    const socket = io();

    let sentence = '';
    let currentPrediction = '';
    let letterTimeout = null;
    const LETTER_DELAY = 3000; // 3 seconds in milliseconds

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => console.error('Error accessing webcam:', err));

    setInterval(() => {
        ctx.drawImage(video, 0, 0, 400, 400);
        const frame = canvas.toDataURL('image/jpeg');
        socket.emit('video_frame', frame);
    }, 100);

    socket.on('prediction', data => {
        const text = data.text;

        // Ignore invalid or empty predictions
        if (!text || text === ' ') return;

        // Clear any existing timeout to avoid overlap
        if (letterTimeout) {
            clearTimeout(letterTimeout);
        }

        // Display the current letter immediately
        currentPrediction = text;
        currentChar.textContent = currentPrediction;

        // Process the letter after 3 seconds
        letterTimeout = setTimeout(() => {
            if (text.length === 1 && /[A-Z]/.test(text)) {
                sentence += text;
                sentenceSpan.textContent = sentence;
            } else if (text === 'Space') {
                sentence += ' ';
                sentenceSpan.textContent = sentence;
            } else if (text === 'Backspace') {
                sentence = sentence.slice(0, -1);
                sentenceSpan.textContent = sentence;
            }

            // Update suggestions (placeholder logic)
            const suggestions = ['word1', 'word2', 'word3', 'word4'];
            suggButtons.forEach((btn, i) => {
                btn.textContent = suggestions[i] || '';
                btn.onclick = () => {
                    if (suggestions[i]) {
                        const lastSpace = sentence.lastIndexOf(' ');
                        sentence = sentence.substring(0, lastSpace + 1) + suggestions[i];
                        sentenceSpan.textContent = sentence;
                    }
                };
            });

            // Clear current character after processing
            currentPrediction = '';
            currentChar.textContent = '';
        }, LETTER_DELAY);
    });

    speakBtn.onclick = () => {
        const utterance = new SpeechSynthesisUtterance(sentence);
        window.speechSynthesis.speak(utterance);
    };

    clearBtn.onclick = () => {
        sentence = '';
        currentPrediction = '';
        sentenceSpan.textContent = '';
        currentChar.textContent = '';
        suggButtons.forEach(btn => btn.textContent = '');
        if (letterTimeout) {
            clearTimeout(letterTimeout);
        }
    };

    // Get the copy button and sentence element
const copyButton = document.getElementById('copy-btn');
const sentenceElement = document.getElementById('sentence');

// Function to copy text to clipboard
function copyTextToClipboard() {
    const textToCopy = sentenceElement.textContent; // Get the current sentence text
    
    if (textToCopy) {
        navigator.clipboard.writeText(textToCopy)  // Copy text to clipboard
            .then(() => {
                alert('Text copied to clipboard!');
            })
            .catch(err => {
                alert('Failed to copy text: ' + err);
            });
    } else {
        alert('No text available to copy.');
    }
}

// Attach the copy function to the button click
copyButton.addEventListener('click', copyTextToClipboard);
