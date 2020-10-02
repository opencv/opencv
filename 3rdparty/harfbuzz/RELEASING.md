HarfBuzz release walk-through checklist:

1. Open gitk and review changes since last release.

   * `git diff $(git describe | sed 's/-.*//').. src/*.h` prints all public API
     changes.

     Document them in NEWS.  All API and API semantic changes should be clearly
     marked as API additions, API changes, or API deletions.  Document
     deprecations.  Ensure all new API / deprecations are in listed correctly in
     docs/harfbuzz-sections.txt.  If release added new API, add entry for new
     API index at the end of docs/harfbuzz-docs.xml.

     If there's a backward-incompatible API change (including deletions for API
     used anywhere), that's a release blocker.  Do NOT release.

2. Based on severity of changes, decide whether it's a minor or micro release
   number bump,

3. Search for REPLACEME on the repository and replace it with the chosen version
   for the release.

4. Make sure you have correct date and new version at the top of NEWS file.

5. Bump version in line 3 of meson.build and configure.ac.
   Do a `meson test -Cbuild` so it both checks the tests and updates
   hb-version.h (use `git diff` to see if is really updated).

6. Commit NEWS, meson.build, configure.ac, and src/hb-version.h, as well as any REPLACEME
   changes you made.  The commit message is simply the release number.  Eg. "1.4.7"

7. Do a `meson dist -Cbuild` that runs the tests against the latest commited changes.
   If doesn't pass, something fishy is going on, reset the repo and start over.

8. Tag the release and sign it: Eg. "git tag -s 1.4.7 -m 1.4.7".  Enter your
   GPG password.

9. Build win32 bundle.  See [README.mingw.md](README.mingw.md).

10. Push the commit and tag out: "git push --follow-tags".

11. Go to GitHub release page [here](https://github.com/harfbuzz/harfbuzz/releases),
    edit the tag, upload win32 bundle and NEWS entry and save.
    No need to upload source tarball as we rely to GitHub's automatic tar.gz generation.
