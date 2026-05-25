package com.PersonaAi.controller;

import com.PersonaAi.dto.LoginRequest;
import com.PersonaAi.dto.RegisterRequest;
import com.PersonaAi.entity.User;
import com.PersonaAi.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    @Value("${personaai.app.url}")
    private String personaaiAppUrl;

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @PostMapping("/register")
    public ResponseEntity<?> registerUser(@RequestBody RegisterRequest request) {
        // בדיקה האם המשתמש או האימייל כבר קיימים
        if (userRepository.existsByUsername(request.getUsername())) {
            return ResponseEntity.badRequest().body("Error: Username is already taken!");
        }
        if (userRepository.existsByEmail(request.getEmail())) {
            return ResponseEntity.badRequest().body("Error: Email is already in use!");
        }

        // יצירת משתמש חדש והצפנת הסיסמה
        User user = new User();
        user.setUsername(request.getUsername());
        user.setEmail(request.getEmail());
        user.setPassword(passwordEncoder.encode(request.getPassword()));

        userRepository.save(user);

        return ResponseEntity.ok("User registered successfully!");
    }

    @PostMapping("/login")
    public ResponseEntity<?> authenticateUser(@RequestBody LoginRequest loginRequest) {
        Optional<User> userOptional = userRepository.findByUsername(loginRequest.getUsername());

        if (userOptional.isPresent() && passwordEncoder.matches(loginRequest.getPassword(), userOptional.get().getPassword())) {
            User user = userOptional.get(); // Fetching the user object
        
            // Changed map type to Object to support Long/Integer IDs
            Map<String, Object> response = new HashMap<>(); 
            response.put("message", "Login successful");
            response.put("userId", user.getId());          // <-- Crucial: Return the actual User ID!
            response.put("username", user.getUsername());  // Optional: Useful for UI welcome messages
            response.put("redirectUrl", personaaiAppUrl);
        
            return ResponseEntity.ok(response);
        } else {
            return ResponseEntity.status(401).body("Invalid username or password");
        }
    }
}