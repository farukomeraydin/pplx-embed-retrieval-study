### contextual vs standard on scifact (3 sentences/chunk)

| system | nDCG@10 [95% CI] | recall@100 |
|---|---|---|
| standard, whole doc | 0.726 [0.683, 0.769] | 0.960 |
| standard, chunks (max-pool) | 0.736 [0.694, 0.778] | 0.979 |
| contextual, chunks (max-pool) | 0.684 [0.638, 0.731] | 0.928 |
| contextual chunks, std-model queries (max-pool) | 0.741 [0.698, 0.782] | 0.956 |

contextual (ctx queries) minus standard (chunks), nDCG@10: -0.053 [-0.081, -0.026]
contextual (std queries) minus standard (chunks), nDCG@10: +0.004 [-0.018, +0.026]

Biggest wins for standard chunks (inspect these):
- [-1.00] Homozygous deletion of murine Sbds gene from osterix-expressing mesenchymal stem and progenitor cells (MPCs) p
- [-1.00] Integrating classroom-based collaborative learning with Web-based collaborative learning leads to subpar class
- [-1.00] Podocytes are motile and migrate in the presence of injury.
- [-0.70] Infection of human T-cell lymphotropic virus type 1 is most frequent in individuals of African origin.
- [-0.68] p16INK4A accumulation is  linked to an abnormal wound response caused by the microinvasive step of advanced Or

Biggest wins for contextual chunks (inspect these):
- [+1.00] Albendazole is used to treat lymphatic filariasis.
- [+1.00] The amount of publicly available DNA data doubles every 10 years.
- [+0.70] Ivermectin is used to treat lymphatic filariasis.
- [+0.68] Cells undergoing methionine restriction may activate miRNAs.
- [+0.61] Epidemiological disease burden from noncommunicable diseases is more prevalent in low economic settings.