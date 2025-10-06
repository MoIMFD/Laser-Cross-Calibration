from datetime import datetime
from itertools import chain

from pydantic import BaseModel, Field

from laser_cross_calibration import __version__
from laser_cross_calibration.gcode.command import BaseCommand, CommentCommand

DEFAULT_HEADER = [
    CommentCommand(comment=f"GCode create by ... Version {__version__}"),
    CommentCommand(
        comment=f"created at: {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}"
    ),
]

DEFAULT_FOOTER = [
    CommentCommand(comment="<" + " End of Program ".center(78, "=") + ">")
]


class Program(BaseModel):
    name: str
    commands: list[BaseCommand] = Field(default_factory=list)

    def render(self) -> str:
        program = chain(
            [CommentCommand(comment="<" + f" {self.name} ".center(78, "=") + ">")],
            DEFAULT_HEADER,
            self.commands,
            DEFAULT_FOOTER,
        )
        return "\n".join(map(str, program))

    def extend_by_program(self, program: "Program"):
        self.extend_by_command(
            CommentCommand(
                comment=f"| insert subprogram {program.name} |".center(80, "-")
            )
        )
        self.commands += program.commands
        self.extend_by_command(CommentCommand(comment="-" * 80))
        return self

    def extend_by_command(self, command: BaseCommand):
        self.commands.append(command)
        return self

    def __add__(self, other: "Program | BaseCommand") -> "Program":
        if isinstance(other, Program):
            self.extend_by_program(program=other)
        elif isinstance(other, BaseCommand):
            self.extend_by_command(command=other)
        else:
            raise ValueError()
        return self
