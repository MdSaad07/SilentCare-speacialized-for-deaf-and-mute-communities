const chatPopup = document.getElementById('chatPopup');
const chatbox = document.getElementById('chatbox');
const input = document.getElementById('input');
let recognition = null;

function toggleChat() {
    chatPopup.classList.toggle('active');
}

function appendMessage(sender, message, isBot = false) {
    const div = document.createElement('div');
    div.className = `message ${sender === 'You' ? 'user-message' : 'bot-message'}`;
    
    if (isBot) {
        const textSpan = document.createElement('span');
        textSpan.textContent = `${sender}: ${message}`;
        
        const speakBtn = document.createElement('button');
        speakBtn.textContent = '🔊';
        speakBtn.className = 'speak-btn';
        speakBtn.onclick = () => speak(message.split(' (confidence')[0]);
        
        div.appendChild(textSpan);
        div.appendChild(speakBtn);
    } else {
        div.textContent = `${sender}: ${message}`;
    }
    
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function speak(text) {
    if (!('speechSynthesis' in window)) {
        console.log('Text-to-speech not supported in this browser.');
        return;
    }
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-IN';
    utterance.pitch = 1;
    utterance.rate = 1;
    utterance.volume = 1;
    
    window.speechSynthesis.speak(utterance);
}

function sendMessage() {
    const message = input.value.trim();
    if (!message) return;
    appendMessage('You', message);
    input.value = '';

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        const botResponse = `${data.response} (confidence: ${data.confidence.toFixed(2)}%)`;
        appendMessage('Bot', botResponse, true);
    })
    .catch(error => {
        appendMessage('Bot', 'Oops, something went wrong!', true);
        console.error('Error:', error);
    });
}

function startVoice() {
    if (!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
        appendMessage('System', 'Sorry, your browser does not support speech recognition.');
        return;
    }

    if (recognition && recognition.isRunning) {
        appendMessage('System', 'Voice input is already running.');
        return;
    }

    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = true;

    recognition.onstart = () => {
        appendMessage('System', 'Listening...');
        console.log('Recognition started');
    };

    recognition.onresult = (event) => {
        const message = event.results[event.results.length - 1][0].transcript.trim();
        appendMessage('You', message);
        console.log(`Heard: "${message}"`);

        if (message.toLowerCase() === 'quit') {
            stopVoice();
            appendMessage('System', 'Voice input stopped.');
            return;
        }

        fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            const botResponse = `${data.response} (confidence: ${data.confidence.toFixed(2)}%)`;
            appendMessage('Bot', botResponse, true);
        })
        .catch(error => {
            appendMessage('Bot', 'Oops, something went wrong!', true);
            console.error('Error:', error);
        });
    };

    recognition.onerror = (event) => {
        appendMessage('System', 'Error: ' + event.error);
        console.log(`Error: ${event.error}`);
        if (event.error === 'no-speech' || event.error === 'aborted') {
            if (recognition.isRunning) {
                recognition.start();
            }
        }
    };

    recognition.onend = () => {
        console.log('Recognition ended');
        if (recognition.isRunning) {
            console.log('Restarting recognition...');
            recognition.start();
        } else {
            console.log('Recognition fully stopped');
        }
    };

    recognition.isRunning = true;
    recognition.start();
}

function stopVoice() {
    if (recognition && recognition.isRunning) {
        recognition.isRunning = false;
        recognition.stop();
        recognition = null;
        console.log('Stopped voice input');
    }
}

input.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});