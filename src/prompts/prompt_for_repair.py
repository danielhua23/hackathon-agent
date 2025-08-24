prompt = """You are a code repair expert. Your task is to analyze failed code, understand the error, and generate a corrected version based on a high-level plan.

**Failed Code:**
```python
{solution}
```

**Test Failure Log:**
```
{test_result}
```

**High-level Correction Plan:**
```markdown
{reflection}
```

**Your Task:**
1.  Read the **Failed Code**, **Test Failure Log**, and **Correction Plan** carefully.
2.  Implement the corrections described in the plan.
3.  Ensure the new code is complete, syntactically correct, and directly addresses the error.
4.  Output **only** the complete, corrected Python code block. Do not add any explanations or text outside the code block.

**Corrected Code:**
"""