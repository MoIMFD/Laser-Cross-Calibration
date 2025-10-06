"""Test G-code command creation and rendering."""

import pytest
from src.laser_cross_calibration.gcode import (
    RAPID_MOVE,
    LINEAR_MOVE,
    PAUSE,
    COMMENT,
    SET_PIN,
    GCodeCommandType,
    GCodeProgram,
)


class TestBasicCommands:
    """Test basic G-code command creation."""

    def test_rapid_move_creation(self):
        """Test RAPID_MOVE command creation."""
        cmd = RAPID_MOVE(x=10, y=20, z=5)
        assert cmd.render() == "G0 X10 Y20 Z5"

    def test_rapid_move_partial(self):
        """Test RAPID_MOVE with partial parameters."""
        cmd = RAPID_MOVE(x=10)
        assert cmd.render() == "G0 X10"

    def test_linear_move_creation(self):
        """Test LINEAR_MOVE command creation."""
        cmd = LINEAR_MOVE(x=30, y=40, f=100)
        assert cmd.render() == "G1 X30 Y40 F100"

    def test_linear_move_with_feed_rate(self):
        """Test LINEAR_MOVE with feed rate."""
        cmd = LINEAR_MOVE(z=10, f=50)
        assert cmd.render() == "G1 Z10 F50"

    def test_comment_creation(self):
        """Test COMMENT command creation."""
        cmd = COMMENT(comment="This is a test comment")
        assert cmd.render() == "; This is a test comment"

    def test_pause_seconds_only(self):
        """Test PAUSE command with seconds only."""
        cmd = PAUSE(s=2.5)
        assert cmd.render() == "G4 S2.5"

    def test_pause_milliseconds_only(self):
        """Test PAUSE command with milliseconds only."""
        cmd = PAUSE(p=1000)
        assert cmd.render() == "G4 P1000"


class TestCommandValidation:
    """Test G-code command parameter validation."""

    def test_comment_requires_comment_param(self):
        """Test that COMMENT requires comment parameter."""
        with pytest.raises(ValueError, match="Missing required parameters"):
            COMMENT()

    def test_pause_requires_time_param(self):
        """Test that PAUSE requires either S or P parameter."""
        with pytest.raises(ValueError, match="Pause command must have either S"):
            cmd = PAUSE()
            cmd.render()

    def test_invalid_parameters_rejected(self):
        """Test that invalid parameters are rejected."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            RAPID_MOVE(invalid_param=123)

    def test_none_values_ignored(self):
        """Test that None values are ignored in rendering."""
        cmd = RAPID_MOVE(x=10, y=None, z=5)
        assert cmd.render() == "G0 X10 Z5"


class TestPauseCommandPrecedence:
    """Test PAUSE command S/P precedence behavior."""

    def test_pause_both_parameters_included(self):
        """Test that both S and P parameters are included in output."""
        cmd = PAUSE(s=3.0, p=2000)
        result = cmd.render()
        assert "S3.0" in result
        assert "P2000" in result
        assert result == "G4 S3.0 P2000"

    def test_pause_s_only(self):
        """Test PAUSE with S parameter only."""
        cmd = PAUSE(s=1.5)
        assert cmd.render() == "G4 S1.5"

    def test_pause_p_only(self):
        """Test PAUSE with P parameter only."""
        cmd = PAUSE(p=500)
        assert cmd.render() == "G4 P500"


class TestGCodeProgram:
    """Test GCodeProgram functionality."""

    def test_empty_program(self):
        """Test empty program creation."""
        program = GCodeProgram()
        assert program.render() == ""

    def test_program_with_commands(self):
        """Test program with multiple commands."""
        program = GCodeProgram(
            commands=[
                COMMENT(comment="Test program"),
                RAPID_MOVE(x=0, y=0, z=0),
                LINEAR_MOVE(x=100, f=200),
                PAUSE(s=1.0),
            ]
        )

        expected = "; Test program\nG0 X0 Y0 Z0\nG1 X100 F200\nG4 S1.0"
        assert program.render() == expected

    def test_add_command_to_program(self):
        """Test adding commands to program."""
        program = GCodeProgram()
        program.add_command(RAPID_MOVE(x=10, y=20))
        program.add_command(LINEAR_MOVE(x=30, f=100))

        expected = "G0 X10 Y20\nG1 X30 F100"
        assert program.render() == expected

    def test_program_with_mixed_commands(self):
        """Test program with various command types."""
        program = GCodeProgram()
        program.add_command(COMMENT(comment="Starting program"))
        program.add_command(RAPID_MOVE(x=0, y=0, z=5))
        program.add_command(LINEAR_MOVE(x=50, y=50, f=150))
        program.add_command(PAUSE(p=500))
        program.add_command(COMMENT(comment="Program complete"))

        expected = (
            "; Starting program\n"
            "G0 X0 Y0 Z5\n"
            "G1 X50 Y50 F150\n"
            "G4 P500\n"
            "; Program complete"
        )
        assert program.render() == expected


class TestInlineComments:
    """Test inline comment functionality."""

    def test_rapid_move_with_comment(self):
        """Test RAPID_MOVE with inline comment."""
        cmd = RAPID_MOVE(x=10, y=20, comment="move to start")
        assert cmd.render() == "G0 X10 Y20 ; move to start"

    def test_linear_move_with_comment(self):
        """Test LINEAR_MOVE with inline comment."""
        cmd = LINEAR_MOVE(x=30, f=100, comment="linear move")
        assert cmd.render() == "G1 X30 F100 ; linear move"

    def test_pause_with_comment(self):
        """Test PAUSE with inline comment."""
        cmd = PAUSE(s=2.0, comment="wait 2 seconds")
        assert cmd.render() == "G4 S2.0 ; wait 2 seconds"

    def test_set_pin_with_comment(self):
        """Test SET_PIN with inline comment."""
        cmd = SET_PIN(p=67, s=255, comment="trigger camera")
        assert cmd.render() == "M42 P67 S255 ; trigger camera"

    def test_comment_command_no_inline(self):
        """Test that COMMENT command doesn't get inline comments."""
        cmd = COMMENT(comment="test comment")
        assert cmd.render() == "; test comment"

    def test_command_without_comment(self):
        """Test that commands without comments work normally."""
        cmd = RAPID_MOVE(x=10, y=20)
        assert cmd.render() == "G0 X10 Y20"

    def test_empty_comment_ignored(self):
        """Test that empty comments are ignored."""
        cmd = RAPID_MOVE(x=10, comment="")
        assert cmd.render() == "G0 X10"

    def test_none_comment_ignored(self):
        """Test that None comments are ignored."""
        cmd = RAPID_MOVE(x=10, comment=None)
        assert cmd.render() == "G0 X10"


class TestCallableAPI:
    """Test the __call__ API for command creation."""

    def test_call_syntax_works(self):
        """Test that __call__ syntax works for all commands."""
        rapid = RAPID_MOVE(x=10, y=20)
        linear = LINEAR_MOVE(x=30, f=100)
        pause = PAUSE(s=2.0)
        comment = COMMENT(comment="test")

        assert rapid.render() == "G0 X10 Y20"
        assert linear.render() == "G1 X30 F100"
        assert pause.render() == "G4 S2.0"
        assert comment.render() == "; test"

    def test_call_vs_create_instance_equivalent(self):
        """Test that __call__ and create_instance are equivalent."""
        cmd1 = RAPID_MOVE(x=10, y=20, z=5)
        cmd2 = RAPID_MOVE.create_instance(x=10, y=20, z=5)

        assert cmd1.render() == cmd2.render()
        assert cmd1.params == cmd2.params
        assert cmd1.command.type == cmd2.command.type
