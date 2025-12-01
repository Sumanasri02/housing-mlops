from pydantic import BaseModel

class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int

    mainroad_no: int
    mainroad_yes: int

    guestroom_no: int
    guestroom_yes: int

    basement_no: int
    basement_yes: int

    hotwaterheating_no: int
    hotwaterheating_yes: int

    airconditioning_no: int
    airconditioning_yes: int

    prefarea_no: int
    prefarea_yes: int

    furnishingstatus_furnished: int
    furnishingstatus_semi_furnished: int
    furnishingstatus_unfurnished: int
