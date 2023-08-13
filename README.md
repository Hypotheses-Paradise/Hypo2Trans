# Hypo2Trans
Single-blind supplementary materials for NeurIPS 2023 submission

| Test   Set  | Baseline | LM $_{rank}$ | T5-${ft}$  | LLaMA-${ft}$ | T5-$LoRA$ | LLaMA-$LoRA$   | $o_{nb}$   | $o_{cp}$ |
|-------------|----------|---------|--------|-------|----------|-------|--------|------|
| WSJ         | $4.5$      | $4.3_{\textcolor{teal}{-4.4\%}}$     | $4.0$      | $x$     | $2.7_{-40.0\%}$      | $\textbf{2.2}_{\textcolor{teal}{-51.1\%}}$   | $4.1$    | $1.2$  |
| ATIS        | $8.3$      | $6.9_{\textcolor{teal}{-16.9\%}}$     | $2.7$    | $x$     | $\textbf{1.7}_{-79.5\%}$      | $1.9_{\textcolor{teal}{-77.1\%}}$   | $5.2$    | $1.1$  |
| CHiME-4     | $11.1$     | $11.0_{\textcolor{teal}{-0.9\%}}$      | $7.9$    | $x$     | $7.0_{-36.9\%}$        | $\textbf{6.6}_{\textcolor{teal}{-40.5\%}}$   | $9.1$    | $2.8$  |
| Tedlium-3   | $8.5$      | $8.0_{\textcolor{teal}{-5.8\%}}$       | $6.6$    | $x$     | $7.4_{-12.9\%}$      | $\textbf{4.6}_{\textcolor{teal}{-45.9\%}}$   | $3.0$      | $0.7$  |
| CV-$accent$   | $14.8$     | $16.0_{\textcolor{gray}{+8.1\%}}$      | $12.9$   | $x$     | $11.0_{-25.7\%}$       | $\textbf{11.0}_{\textcolor{teal}{-25.7\%}}$    | $11.4$   | $7.9$  |
| SwitchBoard | $15.7$     | $15.4_{\textcolor{teal}{-1.9\%}}$    | $15.9$   | $x$     | $14.9_{-5.1\%}$     | $\textbf{14.1}_{\textcolor{teal}{-10.2\%}}$  | $12.6$   | $4.2$  |
| LRS2        | $10.1$     | $9.6_{\textcolor{teal}{-5.0\%}}$     | $9.5$    | $x$     | $\textbf{6.6}_{-34.7\%}$      | $8.8_{\textcolor{teal}{-12.9\%}}$   | $6.9$    | $2.6$  |
| CORAAL      | $21.4$     | $21.4_{\textcolor{teal}{-0\%}}$    | $23.1$   | $x$    | $20.9_{-2.3\%}$     | $\textbf{19.2}_{\textcolor{teal}{-10.3\%}}$  | $21.8$   | $10.7$ |

[Form](https://forms.gle/8p4TVbZXbfHPtqaQA) to request fine-tuning code along with terms of use agreement to prevent malicious uses.
