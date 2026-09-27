# Demonstration Control-System Troubleshooting Notes

This fictional document is provided only for software demonstration.

## Sensor validation

Before diagnosing equipment from a single measurement, compare redundant sensors where available and review calibration status. Check for frozen values, sudden steps, implausible rates of change, and disagreement with related measurements. A sensor-quality flag should prevent an unreliable value from driving an automated recommendation.

## Alarm investigation

Start with the first-out event and reconstruct the event sequence using synchronised timestamps. Review the operating state before the alarm, control commands, interlocks, permissives, and process responses. Avoid treating every downstream alarm as a separate root cause. The earliest credible deviation is usually the most useful starting point.

## Safe use of AI recommendations

An AI assistant may retrieve relevant procedures, summarise evidence, or prioritise likely areas for investigation. It must not replace approved operating limits, protection logic, or an authorised engineer's decision. The interface should show the source passage used for each recommendation and state when the available documents do not contain an answer.

