#!/usr/bin/env bats

# Use this to test requirements.sh
# $ make build_requirements
# $ make test_requirements

@test "Invoking requirements does not throw an error" {
  run ./requirements.sh
  [ "$status" -eq 0 ]
}
