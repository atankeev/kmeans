// Publication is deliberately fail-closed. Only merged Release Please PRs qualify.
const manifestPath = '.release-please-manifest.json';
const stableVersion = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const workflowPaths = ['.github/workflows/go.yml', '.github/workflows/vulnerability.yml'];

async function pendingPulls({github, context}) {
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...context.repo, state: 'closed', labels: 'autorelease: pending', per_page: 100,
  });
  const pulls = [];
  for (const issue of issues.filter(issue => issue.pull_request)) {
    const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: issue.number});
    if (pr.merged_at) pulls.push(pr);
  }
  return pulls;
}

async function manifestAt({github, context}, ref) {
  const {data} = await github.rest.repos.getContent({...context.repo, path: manifestPath, ref});
  if (data.type !== 'file' || data.encoding !== 'base64') throw new Error('manifest is not a file');
  return JSON.parse(Buffer.from(data.content, 'base64').toString('utf8'));
}

async function selectCandidate(api) {
  const {github, context} = api;
  const pending = await pendingPulls(api);
  if (pending.length > 1) throw new Error('multiple merged pending release PRs; resolve ambiguity before retrying');
  if (!pending.length) return null;
  const pr = pending[0];
  const repository = `${context.repo.owner}/${context.repo.repo}`;
  if (pr.base.ref !== 'main' || pr.base.repo.full_name !== repository ||
      pr.head.repo?.full_name !== repository || pr.head.ref !== 'release-please--branches--main' ||
      !/^[a-f0-9]{40}$/.test(pr.merge_commit_sha) ||
      pr.labels.some(label => label.name === 'autorelease: tagged') ||
      !pr.body?.startsWith(':robot: I have created a release *beep* *boop*\n---') ||
      !pr.body.includes('This PR was generated with [Release Please](https://github.com/googleapis/release-please).')) {
    throw new Error(`PR #${pr.number} is not a supported Release Please candidate`);
  }
  const sha = pr.merge_commit_sha;
  const manifest = await manifestAt(api, sha);
  const version = manifest['.'];
  if (Object.keys(manifest).length !== 1 || !stableVersion.test(version)) {
    throw new Error('release manifest must contain one stable root version');
  }
  if (pr.title !== `chore(main): release ${version}` ||
      !new RegExp(`^## \\[?${version.replaceAll('.', '\\.')}[\\] (]`, 'm').test(pr.body)) {
    throw new Error('release PR title, notes and manifest versions disagree');
  }
  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo, pull_number: pr.number, per_page: 100,
  });
  if (!files.some(file => file.filename === manifestPath && ['added', 'modified'].includes(file.status)) ||
      !files.some(file => file.filename === 'CHANGELOG.md' && ['added', 'modified'].includes(file.status))) {
    throw new Error('release PR must change the manifest and changelog');
  }
  const {data: commit} = await github.rest.repos.getCommit({...context.repo, ref: sha});
  const previous = await manifestAt(api, commit.parents[0].sha);
  if (previous['.'] === version) throw new Error('release merge did not change the root version');
  const {data: comparison} = await github.rest.repos.compareCommits({
    ...context.repo, base: sha, head: 'main',
  });
  if (!['ahead', 'identical'].includes(comparison.status)) throw new Error('release SHA is not on main');
  return {number: pr.number, sha, version};
}

async function validateCandidate(api, candidate) {
  const {github, context, core} = api;
  let ready = true;
  for (const path of workflowPaths) {
    const {data: workflow} = await github.rest.actions.getWorkflow({
      ...context.repo, workflow_id: path.split('/').pop(),
    });
    if (workflow.path !== path || workflow.state !== 'active') throw new Error(`unexpected workflow: ${path}`);
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...context.repo, workflow_id: workflow.id, head_sha: candidate.sha,
      event: 'push', branch: 'main', per_page: 100,
    });
    const exact = runs.filter(run => run.head_sha === candidate.sha && run.event === 'push' &&
      run.head_branch === 'main' && run.repository.full_name === `${context.repo.owner}/${context.repo.repo}` &&
      run.workflow_id === workflow.id && run.path === path);
    if (exact.some(run => run.status === 'completed' && run.conclusion !== 'success')) {
      throw new Error(`release SHA has unsuccessful ${path} CI; rerun the failed workflow`);
    }
    if (!exact.length || exact.some(run => run.status !== 'completed')) {
      core.info(`Waiting for successful ${path} push CI on ${candidate.sha}`);
      ready = false;
    }
  }
  return ready;
}

async function publishTag(api, candidate) {
  const {github, context} = api;
  const current = await selectCandidate(api);
  if (JSON.stringify(current) !== JSON.stringify(candidate)) throw new Error('pending release candidate changed');
  if (!await validateCandidate(api, candidate)) throw new Error('candidate CI is still pending');
  const tag = `v${candidate.version}`;
  let reference;
  try {
    ({data: reference} = await github.rest.git.getRef({...context.repo, ref: `tags/${tag}`}));
  } catch (error) {
    if (error.status !== 404) throw error;
  }
  if (reference) {
    if (reference.object.type !== 'tag') throw new Error('existing tag is not annotated; preserve it and investigate without retagging');
    let object = reference.object;
    while (object.type === 'tag') {
      const {data} = await github.rest.git.getTag({...context.repo, tag_sha: object.sha});
      object = data.object;
    }
    if (object.type !== 'commit' || object.sha !== candidate.sha) throw new Error('existing tag points to different source; never retag');
    return;
  }
  const {data: annotation} = await github.rest.git.createTag({
    ...context.repo, tag, message: `Release ${tag}`, object: candidate.sha, type: 'commit',
  });
  await github.rest.git.createRef({...context.repo, ref: `refs/tags/${tag}`, sha: annotation.sha});
}

module.exports = {pendingPulls, selectCandidate, validateCandidate, publishTag};
