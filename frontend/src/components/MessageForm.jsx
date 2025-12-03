import React, { useState } from 'react';

const MessageForm = ({ onSendMessage, onGetHistory, onResetChat, currentThreadId }) => {
  const [message, setMessage] = useState('');
  const [threadId, setThreadId] = useState('default');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSendMessage(message, threadId);
    setMessage('');
  };

  const handleGetHistory = () => {
    onGetHistory(threadId);
  };

  const handleResetChat = () => {
    onResetChat(threadId);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="message">Message:</label>
        <textarea
          id="message"
          name="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Enter your message here..."
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="threadId">Thread ID:</label>
        <input
          type="text"
          id="threadId"
          name="threadId"
          value={threadId}
          onChange={(e) => setThreadId(e.target.value)}
          placeholder="Enter thread ID (optional)"
        />
      </div>
      
      <button type="submit">Send Message</button>
      <button type="button" onClick={handleGetHistory}>Get History</button>
      <button type="button" onClick={handleResetChat}>Reset Chat</button>
    </form>
  );
};

export default MessageForm;
