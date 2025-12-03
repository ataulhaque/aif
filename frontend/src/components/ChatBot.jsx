import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';

const API_BASE = 'http://127.0.0.1:8001';

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [threadId, setThreadId] = useState('default');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (content, role, timestamp = new Date()) => {
    const newMessage = {
      id: Date.now(),
      content,
      role,
      timestamp,
      threadId
    };
    setMessages(prev => [...prev, newMessage]);
    return newMessage;
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    setError(null);
    setIsLoading(true);
    
    // Add user message immediately
    addMessage(inputMessage, 'user');
    const userMessage = inputMessage;
    setInputMessage('');

    try {
      const response = await fetch(`${API_BASE}/chat?thread_id=${threadId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: userMessage
        })
      });

      const data = await response.json();

      if (response.ok) {
        // Add AI response
        addMessage(data.response, 'assistant');
      } else {
        setError(`Error: ${data.detail || 'Unknown error'}`);
      }

    } catch (error) {
      setError(`Network Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const getHistory = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/chat/history/${threadId}`);
      const data = await response.json();

      if (response.ok) {
        const historyMessages = data.messages.map((msg, index) => ({
          id: `history-${index}`,
          content: msg.content,
          role: msg.role,
          timestamp: new Date(),
          threadId
        }));
        setMessages(historyMessages);
      } else {
        setError(`Error getting history: ${data.detail || 'Unknown error'}`);
      }

    } catch (error) {
      setError(`Network Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const resetChat = async () => {
    if (!window.confirm(`Are you sure you want to reset the chat for thread "${threadId}"?`)) {
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE}/chat/reset/${threadId}`, {
        method: 'POST'
      });
      const data = await response.json();

      if (response.ok) {
        setMessages([]);
        setError(null);
      } else {
        setError(`Error resetting chat: ${data.detail || 'Unknown error'}`);
      }

    } catch (error) {
      setError(`Network Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="container">
      <div className="chat-header">
        <h1>🤖 AI ChatBot</h1>
      </div>
      
      <div className="chat-body">
        <div className="chat-messages">
          {messages.map((message) => (
            <div key={message.id} className={`message-bubble ${message.role}`}>
              <div>{message.content}</div>
              <div className="message-time">{formatTime(message.timestamp)}</div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message-bubble assistant">
              <div className="loading">AI is thinking...</div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input-area">
          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
          
          <form onSubmit={sendMessage} className="input-form">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              rows="1"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage(e);
                }
              }}
              disabled={isLoading}
            />
            <button 
              type="submit" 
              className="btn-send"
              disabled={isLoading || !inputMessage.trim()}
            >
              ➤
            </button>
          </form>
          
          <div className="control-buttons">
            <input
              type="text"
              value={threadId}
              onChange={(e) => setThreadId(e.target.value)}
              placeholder="Thread ID"
              className="thread-input"
            />
            <button onClick={getHistory} className="btn" disabled={isLoading}>
              📜 History
            </button>
            <button onClick={resetChat} className="btn" disabled={isLoading}>
              🗑️ Reset
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;
