from pydantic import BaseModel, ConfigDict, Field, model_validator

class LabelDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    definition: str

class ProposedLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str


class LabellingOutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assigned_label_ids: list[str] = Field(default_factory=list)
    proposed_labels: list[ProposedLabel] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_some_output(self):
        if not self.assigned_label_ids and not self.proposed_labels:
            raise ValueError(
                "At least one assigned label or proposed label is required"
            )
        return self
    

class LabelledDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    output: LabellingOutputSchema