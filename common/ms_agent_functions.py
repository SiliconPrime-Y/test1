import os

import json
from datetime import datetime, timedelta
import asyncio
from .business_logic import (
    get_customer,
    get_customer_appointments,
    get_customer_orders,
    schedule_appointment,
    get_available_appointment_slots,
    prepare_agent_filler_message,
    prepare_farewell_message,
)

from openai import OpenAI
from dotenv import load_dotenv

from .retrieval_data import IMPROVE_TIME_TABLE, UPSALE_SERVICE, TIME_TABLE

load_dotenv()


async def find_customer(params):
    """Look up a customer by phone, email, or ID."""
    phone = params.get("phone")
    email = params.get("email")
    customer_id = params.get("customer_id")

    result = await get_customer(phone=phone, email=email, customer_id=customer_id)
    return result


async def get_appointments(params):
    """Get appointments for a customer."""
    customer_id = params.get("customer_id")
    if not customer_id:
        return {"error": "customer_id is required"}

    result = await get_customer_appointments(customer_id)
    return result


async def get_orders(params):
    """Get orders for a customer."""
    customer_id = params.get("customer_id")
    if not customer_id:
        return {"error": "customer_id is required"}

    result = await get_customer_orders(customer_id)
    return result


async def create_appointment(params):
    """Schedule a new appointment."""
    customer_id = params.get("customer_id")
    date = params.get("date")
    service = params.get("service")

    if not all([customer_id, date, service]):
        return {"error": "customer_id, date, and service are required"}

    result = await schedule_appointment(customer_id, date, service)
    return result


async def check_availability(params):
    """Check available appointment slots."""
    start_date = params.get("start_date")
    end_date = params.get(
        "end_date", (datetime.fromisoformat(start_date) + timedelta(days=7)).isoformat()
    )

    if not start_date:
        return {"error": "start_date is required"}

    result = await get_available_appointment_slots(start_date, end_date)
    return result


async def is_available(params):
    try:
        date = params.get('date', "").title()
        name = params.get('name', "").title()
        service = params.get('service', "").title()

        time_table = IMPROVE_TIME_TABLE.get(service.title(), TIME_TABLE)

        if date == 'Tomorrow':
            date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%A")
        elif date == 'Today':
            date = datetime.date.today().strftime("%A")

        if not name and not date:
            name = next(iter(time_table), "")

        valid_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}

        if name and name not in time_table and date not in valid_days:
            available_therapists = ", ".join(time_table.keys())
            return json.dumps({
                "check": f"{name} is not available. Available therapists: {available_therapists}. Please provide a specific day."
            })

        if name and name not in time_table and date in valid_days:
            available_therapists = [n for n, schedule in time_table.items() if schedule.get(date) != "Off"]
            if available_therapists:
                return json.dumps({
                    "check": f"Available therapists on {date}: {', '.join(available_therapists)}"
                })
            return json.dumps({"check": "No available therapists on that day."})

        if not name and date in valid_days:
            for therapist, schedule in time_table.items():
                if schedule.get(date) and schedule[date] != "Off":
                    return json.dumps({
                        "check": f"{therapist} is available from {schedule[date]} on {date}."
                    })

        if name and date not in valid_days:
            for available_date, available_time in time_table[name].items():
                if available_time != "Off":
                    return json.dumps({
                        "check": f"{name} is available from {available_time} on {available_date}."
                    })

        if time_table[name].get(date) and time_table[name][date] != "Off":
            return json.dumps({
                "check": f"{name} is available from {time_table[name][date]} on {date}."
            })
        
        other_dates = [f"{name} is available on {d} from {t}" for d, t in time_table[name].items() if t != "Off"]
        other_therapists = [f"{n} is available on {date} from {time_table[n][date]}" for n in time_table if time_table[n].get(date) and time_table[n][date] != "Off"]
        
        return json.dumps({
            "check": f"{name} is off on {date}. Available options: {', '.join(other_dates)}. Or consider {', '.join(other_therapists)}."
        })
    
    except Exception as err:
        print("Error:", err)
        return json.dumps({"check": "An error occurred while processing the request."})


async def find_nearest_location(params):
    print("***************Go to find location***************")
    try:
        client = OpenAI(api_key=os.getenv('OPENAPI_KEY'))
        location = params.get('location')
        if not location:
            return {
                "nearest_location": "Please, Could you provide me more specific address?"
            }

        prompt = f"Find 1 Massage Envy location near {location}. Only get location's name and address."
        completion = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        print("Location: ", completion.choices[0].message.content)

        return {
            "nearest_location": completion.choices[0].message.content
        }
    except Exception as err:
        print("Error: ", err)
        return {"nearest_location": "There are no nearest franchised."}

async def upsale_service(params):
    try:
        service = params.get('service').title()
        if not service:
            return {
                "upsale_service_infor": "Please, Could you provide me more specific address?"
            }
        print("\n".join(UPSALE_SERVICE[service]))

        print("****** End Upsale Service ******* \n")

        return json({
            "upsale_service_infor": "\n".join(UPSALE_SERVICE[service])
        })
    except Exception as err:
        print("Error: ", err)
        return json({"upsale_service_infor": "There are no information detail for this service."})


async def agent_filler(websocket, params):
    """
    Handle agent filler messages while maintaining proper function call protocol.
    """
    result = await prepare_agent_filler_message(websocket, **params)
    return result


async def end_call(websocket, params):
    """
    End the conversation and close the connection.
    """
    farewell_type = params.get("farewell_type", "general")
    result = await prepare_farewell_message(websocket, farewell_type)
    return result


# Function definitions that will be sent to the Voice Agent API
FUNCTION_DEFINITIONS = [
    {
        "name": "agent_filler",
        "description": """Use this function to provide natural conversational filler before looking up information.
        ALWAYS call this function first with message_type='lookup' when you're about to look up customer information.
        After calling this function, you MUST immediately follow up with the appropriate lookup function (e.g., find_customer).""",
        "parameters": {
            "type": "object",
            "properties": {
                "message_type": {
                    "type": "string",
                    "description": "Type of filler message to use. Use 'lookup' when about to search for information.",
                    "enum": ["lookup", "general"],
                }
            },
            "required": ["message_type"],
        },
    },
    {
        "name": "find_nearest_location",
        "description": """
        Massage envy has more 1000 franchised locations in US. find_nearest_location function search 1 Massege Envy locate near user's address.
        Input: user's address
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "user's location",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "is_therapist_availble",
        "description": """
        get therapist's status.
        Collect user require
        - name: technician's name or therapist's name.
        - date: it is a weekday (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday).
        - service's name: 'Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage session', 'Microderm Infusion', 'Oxygenating Treatment', 'Rapid Tension Relief session', 'Total Body Stretch session'.
        """,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "technician's name",
                },
                "date": {
                    "type": "string",
                    "description": "date: it is a weekday (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, Today, Tomorrow)",
                },
                "service": {
                    "type": "string",
                    "description": "- service's name: 'Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage session', 'Microderm Infusion', 'Oxygenating Treatment', 'Rapid Tension Relief session', 'Total Body Stretch session'",
                    "enum": ['Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage session', 'Microderm Infusion', 'Oxygenating Treatment', 'Rapid Tension Relief session', 'Total Body Stretch session'],
                },
            },
            "required": ["name", "date", "service"],
        },
    },
    {
        "name": "enhancement_option",
        "description": """Get information for up sale services:'Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage Session', 'Microderm Infusion', 'Oxygenating Treatment', 'Total body Stretch Session'""",
        "parameters": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": """- service's name: 'Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage session', 'Microderm Infusion', 'Oxygenating Treatment', 'Rapid Tension Relief session', 'Total Body Stretch session'. if not mention, set default value.
                                    Default value: ''""",
                    "enum": ['Chemical Peel', 'Customized Facial', 'Dermaplaning Treatment', 'Massage session', 'Microderm Infusion', 'Oxygenating Treatment', 'Rapid Tension Relief session', 'Total Body Stretch session', ''],
                }
            },
            "required": ["service"],
        },
    },
    {
        "name": "end_call",
        "description": """End the conversation and close the connection. Call this function when:
        - User wants to end the conversation
        
        Examples of triggers:
        - "Thank you, bye!"
        - "Goodbye"
        
        Do not call this function if the user is just saying thanks but continuing the conversation.""",
        "parameters": {
            "type": "object",
            "properties": {
                "farewell_type": {
                    "type": "string",
                    "description": "Type of farewell to use in response",
                    "enum": ["thanks", "general", "help"],
                }
            },
            "required": ["farewell_type"],
        },
    },
]

# Map function names to their implementations
FUNCTION_MAP = {
    "find_nearest_location": find_nearest_location,
    "is_therapist_availble": is_available,
    "enhancement_option": upsale_service,
    "agent_filler": agent_filler,
    "end_call": end_call,
}
