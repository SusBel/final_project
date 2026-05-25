package com.PersonaAi; 

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * PersonaAI Authentication Server
 *
 * Runs on port 8080.  The PersonaAI Python/FastAPI backend runs on port 8000.
 *
 * Flow:
 *   1. User visits http://localhost:8080/login.html  (or /signup.html)
 *   2. Java validates credentials against MySQL → issues a session token
 *   3. Browser stores token in localStorage and is redirected to
 *      http://localhost:8000/index.html  (the PersonaAI chat page)
 */

@SpringBootApplication
public class AuthApplication {

    public static void main(String[] args) {
        SpringApplication.run(AuthApplication.class, args);
    }
}