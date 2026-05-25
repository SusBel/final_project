package com.PersonaAi.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "sessions")
public class Session {

    @Id
    @Column(name = "session_id", length = 36)
    private String sessionId;

    @Column(name = "user_id")
    private Long userId; // יכול להיות ריק אם המשתמש לא מחובר

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "last_updated_at")
    private LocalDateTime lastUpdatedAt = LocalDateTime.now();

    @Column(length = 20)
    private String status = "active";

    @Column(name = "current_state", length = 50)
    private String currentState = "start";

    // בנאי רק, גטרים וסטרים
    public Session() {}

    @PreUpdate
    public void setLastUpdate() {
        this.lastUpdatedAt = LocalDateTime.now();
    }

    // --- Getters and Setters ---
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }
    public Long getUserId() { return userId; }
    public void setUserId(Long userId) { this.userId = userId; }
    public String getCurrentState() { return currentState; }
    public void setCurrentState(String currentState) { this.currentState = currentState; }
}