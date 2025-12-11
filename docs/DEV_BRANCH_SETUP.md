# Neurogen v1.1 Development Branch Setup

## 🎯 Objective
Freeze the current `main` branch at v1.0.0, create a tagged release, and establish a clean development branch for v1.1 work.

---

## 📋 Step-by-Step Instructions

### 1. Freeze and Tag v1.0.0

```bash
# Ensure you're on main and everything is committed
git checkout main
git status  # Should show "nothing to commit, working tree clean"

# If there are uncommitted changes, commit them first
git add .
git commit -m "chore: finalize v1.0.0 release"

# Create an annotated tag for v1.0.0
git tag -a v1.0.0 -m "Release v1.0.0: Initial Neurogen implementation with evolutionary networks and local learning"

# Push the tag to remote
git push origin v1.0.0

# Push main branch (if needed)
git push origin main
```

### 2. Create Development Branch for v1.1

```bash
# Create and switch to new development branch
git checkout -b dev/v1.1

# Push the new branch to remote
git push -u origin dev/v1.1
```

### 3. Verify Setup

```bash
# List all tags
git tag
# Should show: v1.0.0

# List all branches
git branch -a
# Should show: main, dev/v1.1, and remote branches

# Confirm you're on dev/v1.1
git branch
# Should show: * dev/v1.1
```

---

## 🔄 Development Workflow on dev/v1.1

### Daily Development Cycle

```bash
# 1. Start your work session
git checkout dev/v1.1
git pull origin dev/v1.1  # Get latest changes

# 2. Make your changes
# ... edit files, add features, fix bugs ...

# 3. Commit frequently with clear messages
git add <files>
git commit -m "feat: add synthetic dataset generator"
# or
git commit -m "fix: resolve weight initialization issue"
# or
git commit -m "refactor: modularize training loop"

# 4. Push to remote regularly
git push origin dev/v1.1
```

### Commit Message Convention

Use conventional commits for clarity:

- `feat:` - New features
- `fix:` - Bug fixes
- `refactor:` - Code restructuring without behavior change
- `docs:` - Documentation updates
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks (dependencies, configs)
- `perf:` - Performance improvements

**Examples:**
```bash
git commit -m "feat: implement ConfigManager for reproducible runs"
git commit -m "fix: correct Hebbian learning weight update formula"
git commit -m "refactor: extract data loading into separate module"
git commit -m "docs: add architecture diagrams to ARCHITECTURE.md"
```

### Feature Branch Workflow (Optional)

For larger features, create sub-branches:

```bash
# Create feature branch from dev/v1.1
git checkout dev/v1.1
git checkout -b feature/config-system

# Work on feature
# ... make changes ...
git add .
git commit -m "feat: implement YAML config loader"

# Push feature branch
git push -u origin feature/config-system

# When complete, merge back to dev/v1.1
git checkout dev/v1.1
git merge feature/config-system
git push origin dev/v1.1

# Delete feature branch (optional)
git branch -d feature/config-system
git push origin --delete feature/config-system
```

---

## 🚀 When v1.1 is Ready for Release

### Merge dev/v1.1 → main

```bash
# 1. Ensure dev/v1.1 is fully tested and ready
git checkout dev/v1.1
git pull origin dev/v1.1

# 2. Switch to main and merge
git checkout main
git pull origin main
git merge dev/v1.1 --no-ff -m "chore: merge v1.1 development into main"

# 3. Tag the new release
git tag -a v1.1.0 -m "Release v1.1.0: Modular architecture with reproducible training"

# 4. Push everything
git push origin main
git push origin v1.1.0
```

---

## 🛡️ Best Practices

### ✅ Do's
- ✅ Commit frequently with descriptive messages
- ✅ Pull before starting work each day
- ✅ Test your changes before pushing
- ✅ Keep `main` stable and production-ready
- ✅ Use `dev/v1.1` for all v1.1 development
- ✅ Document breaking changes in commit messages

### ❌ Don'ts
- ❌ Don't commit directly to `main` during v1.1 development
- ❌ Don't force push (`git push -f`) to shared branches
- ❌ Don't commit large binary files or datasets
- ❌ Don't leave broken code in `dev/v1.1` overnight
- ❌ Don't merge without testing

---

## 📊 Branch Protection (Optional - GitHub)

If using GitHub, consider protecting `main`:

1. Go to **Settings** → **Branches**
2. Add rule for `main`:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass
   - ✅ Include administrators

This ensures all changes to `main` go through review.

---

## 🔍 Quick Reference

| Command | Purpose |
|---------|---------|
| `git checkout dev/v1.1` | Switch to development branch |
| `git pull origin dev/v1.1` | Get latest changes |
| `git status` | Check current state |
| `git log --oneline -10` | View recent commits |
| `git diff` | See uncommitted changes |
| `git branch -a` | List all branches |
| `git tag` | List all tags |

---

## 📝 Notes

- The `v1.0.0` tag is immutable - it will always point to the initial release
- All v1.1 work happens on `dev/v1.1` until ready for release
- Keep `main` clean and deployable at all times
- Use descriptive branch names for features: `feature/dataset-loader`, `fix/memory-leak`, etc.

---

**Ready to start v1.1 development!** 🚀
