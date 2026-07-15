# scripts/Tools — agent-callable tool scripts

| Script | Role |
|--------|------|
| `kb_retrieve.py` | Local Qdrant body chunks + parent expand |
| `tavily_search.py` | Web discovery (snippet only) |
| `fetch_page.py` | Fetch URL and extract main body |
| `run_local_script.py` | Whitelisted local script runner |
| `mineru_client.py` / `mineru_parse.py` | PDF parse (optional) |

Python:

```powershell
$py = "E:\application\miniforge3\envs\copilot-agent\python.exe"
& $py scripts\Tools\kb_retrieve.py "radio propagation GNN"
& $py scripts\Tools\tavily_search.py "graph neural networks radio map"
& $py scripts\Tools\fetch_page.py "https://example.com"
```

Agent code should prefer `src.tools.registry.ToolRegistry`.
