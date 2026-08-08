graph TD
    A[BDI Agent Start] --> B[__init__]
    B --> C[_initialize_string_desires]
    
    D[bdi_cycle] --> E{Active desires exist<br/>and no intentions?}
    E -->|Yes| F[generate_intentions_from_desires]
    E -->|No| G{Intentions exist?}
    
    F --> F1[Generate high-level intentions]
    F1 --> F3[Update agent with single-step intentions]
    F3 --> G
    
    G -->|Yes| H[execute_intentions]
    G -->|No| I[Skip execution]
    
    H --> H1{Intention complete?}
    H1 -->|Yes| H2[Mark desire achieved<br/>Remove intention]
    H1 -->|No| H3[Execute current step]
    
    H3 --> H4{Tool call or<br/>descriptive step?}
    H4 -->|Tool call| H5[Execute via self.run with tool prompt]
    H4 -->|Descriptive| H6[Execute via self.run with description]
    
    H5 --> H7[_analyze_step_outcome_and_update_beliefs]
    H6 --> H7
    
    H7 --> H7A[_generate_history_context<br/>Get recent step history]
    H7A --> H7B[Enhanced LLM assessment<br/>with historical context]
    H7B --> H8{Step successful?}
    
    H8 -->|Yes| H8A[Record step success<br/>in intention.step_history]
    H8 -->|No| H8B[Record step failure<br/>in intention.step_history]
    
    H8A --> H9[Increment step counter]
    H8B --> H15[Step failed, no intervention]
    
    H9 --> H11{Final step?}
    H11 -->|Yes| H12[Mark desire achieved<br/>Remove intention]
    H11 -->|No| H13[Continue to next step]
    
    H2 --> J
    H12 --> J
    H13 --> J
    H15 --> J
    I --> J
    
    J{Intentions still exist?}
    J -->|Yes| K[_reconsider_current_intention]
    J -->|No| L[End BDI cycle]
    
    K --> K1[Format beliefs and remaining steps]
    K1 --> K1A[_generate_history_context<br/>Get detailed step history<br/>max_history=5, include_details=True]
    K1A --> K2[Enhanced LLM plan assessment<br/>with historical patterns analysis]
    K2 --> K3{Plan still valid?}
    K3 -->|Yes| L
    K3 -->|No| K4[Remove invalid intention<br/>Set desire to PENDING]
    K4 --> L
    
    L --> M[log_states]
    M --> N[BDI Cycle Complete]
    
    %% Exception handling path
    H5 -.->|Exception| EX1[Record exception<br/>in step_history]
    H6 -.->|Exception| EX1
    EX1 --> EX2[Mark desire as FAILED<br/>Remove intention]
    EX2 --> J
    
    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style F fill:#e8f5e8
    style H fill:#fff3e0
    style K fill:#f1f8e9
    style H7A fill:#e3f2fd
    style H8A fill:#e8f5e8
    style H8B fill:#ffebee
    style K1A fill:#e3f2fd
    style EX1 fill:#ffcdd2
    style EX2 fill:#ffcdd2
