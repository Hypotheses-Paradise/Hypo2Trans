# Hypo2Trans
Single-blind supplementary materials for NeurIPS 2023 submission


The table below presents the WER(%) results of H2T-*ft* and H2T-*LoRA* in finetuning setting, where $o_{nb}$ and $o_{cp}$ respectively denote n-best oracle and compositional oracle:
| Test   Set  | Baseline | LM $_{rank}$ | T5-*ft*  | LLaMA-*ft* | T5-*LoRA* | LLaMA-*LoRA*   | $o_{nb}$   | $o_{cp}$ |
|-------------|----------|---------|--------|-------|----------|-------|--------|------|
| WSJ         | 4.5      | 4.3<sub>-4.4%</sub>     | 4.0<sub>-11.1%</sub>      |   3.8<sub>-15.6%</sub>   | 2.7<sub>-40.0%</sub>      | **2.2<sub>-51.1%</sub>**   | 4.1    | 1.2  |
| ATIS        | 8.3      | 6.9<sub>-16.9%</sub>     | 2.7<sub>-67.5%</sub>    |   3.4<sub>-59.0%</sub>   | **1.7<sub>-79.5%</sub>**      | 1.9<sub>-77.1%</sub>   | 5.2    | 1.1  |
| CHiME-4     | 11.1     | 11.0<sub>-0.9%</sub>      | 7.9<sub>-28.8%</sub>    |   8.2<sub>-26.1%</sub>   | 7.0<sub>-36.9%</sub>        | **6.6<sub>-40.5%</sub>**   | 9.1    | 2.8  |
| Tedlium-3   | 8.5      | 8.0<sub>-5.8%</sub>       | 6.6<sub>-22.4%</sub>    |   5.2<sub>-38.8%</sub>   | 7.4<sub>-12.9%</sub>      | **4.6<sub>-45.9%</sub>**   | 3.0      | 0.7  |
| CV-accent   | 14.8     | 16.0<sub>+8.1%</sub>      | 12.9<sub>-12.8%</sub>   |   15.5<sub>+4.7%</sub>   | 11.0<sub>-25.7%</sub>       | **11.0<sub>-25.7%</sub>**    | 11.4   | 7.9  |
| SwitchBoard | 15.7     | 15.4<sub>-1.9%</sub>    | 15.9<sub>+1.3%</sub>   |  18.4<sub>+17.1%</sub>   | 14.9<sub>-5.1%</sub>     | **14.1<sub>-10.2%</sub>**  | 12.6   | 4.2  |
| LRS2        | 10.1     | 9.6<sub>-5.0%</sub>     | 9.5<sub>-5.9%</sub>    |   10.2<sub>+1.0%</sub>   | **6.6<sub>-34.7%</sub>**      | 8.8<sub>-12.9%</sub>   | 6.9    | 2.6  |
| CORAAL      | 21.4     | 21.4<sub>-0.0%</sub>    | 23.1<sub>+7.9%</sub>   |   22.9<sub>+7.0%</sub>  | 20.9<sub>-2.3%</sub>     | **19.2<sub>-10.3%</sub>**  | 21.8   | 10.7 |

[Form](https://forms.gle/8p4TVbZXbfHPtqaQA) to request fine-tuning code along with terms of use agreement to prevent malicious uses.
