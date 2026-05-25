package com.PersonaAi.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "logs")
public class ChatLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "session_id", length = 36)
    private String sessionId;

    @Column(columnDefinition = "TEXT")
    private String message;

    private String intent;
    private String emotion;
    private String action;
    
    @Column(name = "prev_state", length = 50)
    private String prevState;
    
    @Column(name = "next_state", length = 50)
    private String nextState;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt = LocalDateTime.now();

    public ChatLog() {}
    // תוכל להוסיף Getters ו-Setters בהמשך במידת הצורך
}