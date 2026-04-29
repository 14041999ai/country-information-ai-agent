document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const questionInput = document.getElementById('question-input');
    const chatContainer = document.getElementById('chat-container');
    const sendButton = document.getElementById('send-button');

    // Add a new message to the chat
    function addMessage(text, isUser = false, meta = null) {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper ${isUser ? 'user' : 'bot'}`;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        
        // Handle newlines in bot responses
        const formattedText = text.replace(/\n/g, '<br>');
        messageDiv.innerHTML = formattedText;

        // Add metadata if provided (e.g., country, fields used)
        if (meta && (meta.country || meta.fields)) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            
            if (meta.country) {
                const countryTag = document.createElement('span');
                countryTag.className = 'meta-tag';
                countryTag.textContent = meta.country;
                metaDiv.appendChild(countryTag);
            }
            
            if (meta.fields && meta.fields.length > 0) {
                const fieldsTag = document.createElement('span');
                fieldsTag.className = 'meta-tag';
                fieldsTag.textContent = meta.fields.join(', ');
                metaDiv.appendChild(fieldsTag);
            }
            
            messageDiv.appendChild(metaDiv);
        }

        wrapper.appendChild(messageDiv);
        chatContainer.appendChild(wrapper);
        scrollToBottom();
    }

    // Show typing indicator
    function showTyping() {
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper bot typing-indicator-wrapper';
        wrapper.id = 'typing-indicator';
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot-message';
        
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'typing-indicator';
        indicatorDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        
        messageDiv.appendChild(indicatorDiv);
        wrapper.appendChild(messageDiv);
        chatContainer.appendChild(wrapper);
        scrollToBottom();
    }

    // Remove typing indicator
    function removeTyping() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Scroll to the bottom of the chat container
    function scrollToBottom() {
        chatContainer.scrollTo({
            top: chatContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const question = questionInput.value.trim();
        if (!question) return;

        // Disable input and button
        questionInput.value = '';
        questionInput.disabled = true;
        sendButton.disabled = true;

        // Add user message
        addMessage(question, true);
        
        // Show typing indicator
        showTyping();

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            const data = await response.json();
            
            removeTyping();
            
            if (response.ok) {
                const meta = {
                    country: data.country,
                    fields: data.fields_used
                };
                addMessage(data.answer, false, meta);
            } else {
                addMessage(`Error: ${data.detail || 'Failed to get an answer.'}`, false);
            }
        } catch (error) {
            removeTyping();
            addMessage(`Connection error: ${error.message}`, false);
        } finally {
            // Re-enable input and button
            questionInput.disabled = false;
            sendButton.disabled = false;
            questionInput.focus();
        }
    });

    // Initial focus
    questionInput.focus();
});
