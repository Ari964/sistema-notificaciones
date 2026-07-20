CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    age             INTEGER NOT NULL,
    gender          TEXT NOT NULL,
    medication_type TEXT NOT NULL,
    dosage          REAL NOT NULL,
    education_level TEXT NOT NULL,
    income_level    TEXT NOT NULL,
    social_support  TEXT NOT NULL,
    disease_severity TEXT NOT NULL,
    num_comorbidities INTEGER NOT NULL,
    insurance_coverage TEXT NOT NULL,
    previous_adherence INTEGER NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id            INTEGER NOT NULL,
    adherence_probability REAL NOT NULL,
    adherence_class       TEXT NOT NULL,
    risk_level            TEXT NOT NULL,
    notification_strategy TEXT NOT NULL,
    shap_values           TEXT,
    model_version         TEXT DEFAULT 'cnn-gru-v2.0',
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS notifications (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id          INTEGER NOT NULL,
    prediction_id       INTEGER NOT NULL,
    notification_type   TEXT NOT NULL,
    channel             TEXT,
    message             TEXT NOT NULL,
    status              TEXT DEFAULT 'pendiente',
    max_retries         INTEGER DEFAULT 1,
    escalation          TEXT,
    sent_at             TIMESTAMP,
    acknowledged_at     TIMESTAMP,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id),
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id   INTEGER NOT NULL,
    action      TEXT NOT NULL,
    details     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient ON predictions(patient_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_patient ON notifications(patient_id, status);
CREATE INDEX IF NOT EXISTS idx_notifications_pending ON notifications(status) WHERE status = 'pendiente';
CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_log(entity_type, entity_id);
