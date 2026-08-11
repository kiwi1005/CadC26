# Use exact cap margin for development decisions

HCFP-5090 uses exact uncapped `J` and cap margin to train, rank, diagnose, and compare candidates, while official capped cost remains the promotion and release measure. Capped cost hides useful ordering for 56 changed Q6 cases, and the internal `cost < 9.99` competitive label is not equivalent to crossing the official cap.
