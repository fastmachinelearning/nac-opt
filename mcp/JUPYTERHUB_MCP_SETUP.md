# Claude Code + SNAC-Pack MCP on JupyterHub

Copy-paste these into the JupyterHub terminal, in order.

## 1. Install Claude Code

```bash
npm config set prefix ~/.npm-global
npm install -g @anthropic-ai/claude-code
```

## 2. Install the MCP dependency

```bash
pip install --user fastmcp
```

## 3. Set up PATH

```bash
echo 'export PATH=$HOME/.npm-global/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
export PATH=$HOME/.npm-global/bin:$HOME/.local/bin:$PATH
```

## 4. Register the MCP server

```bash
chmod +x ~/nac-opt/mcp/launch_nac_opt_mcp.sh
claude mcp remove nac-opt 2>/dev/null
claude mcp add nac-opt -- ~/nac-opt/mcp/launch_nac_opt_mcp.sh
claude mcp list
```

You should see:

```
nac-opt: /home/jovyan/nac-opt/mcp/launch_nac_opt_mcp.sh  - ✓ Connected
```

## 5. Start Claude

```bash
cd ~/nac-opt
claude
```

Then run `/login` — it prints a URL. Open it in your laptop browser, approve, and
paste the code back into the terminal. Run `/mcp` inside Claude to confirm the
`nac-opt` tools are available.

---

### If something goes wrong

- **`✓ Connected` doesn't show / `fastmcp: command not found`** — re-run step 3, then step 4.
- **`claude: command not found` after a pod restart** — re-run step 3.
